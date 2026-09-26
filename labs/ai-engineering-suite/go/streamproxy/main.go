// A bounded, cancellation-aware SSE proxy. No external Go dependencies.
package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

const maxBody = 1 << 20
const maxStream = 16 << 20

// Parser accepts LF, CRLF, and CR; incomplete EOF events are never dispatched.
type Parser struct {
	line, data []byte
	cr         bool
	first      bool
	OnEvent    func([]byte)
}

func (p *Parser) newline() {
	line := p.line
	if !p.first {
		line = bytes.TrimPrefix(line, []byte{0xef, 0xbb, 0xbf})
		p.first = true
	}
	if len(line) == 0 {
		if len(p.data) > 0 {
			p.OnEvent(bytes.TrimSuffix(p.data, []byte{'\n'}))
		}
		p.data = nil
	} else if bytes.HasPrefix(line, []byte("data:")) {
		value := bytes.TrimPrefix(line[5:], []byte{' '})
		p.data = append(p.data, value...)
		p.data = append(p.data, '\n')
	} else if bytes.Equal(line, []byte("data")) {
		p.data = append(p.data, '\n')
	}
	p.line = nil
}

func (p *Parser) Feed(chunk []byte) error {
	for _, b := range chunk {
		if p.cr && b == '\n' {
			p.cr = false
			continue
		}
		p.cr = false
		if b == '\r' || b == '\n' {
			p.newline()
			p.cr = b == '\r'
		} else {
			p.line = append(p.line, b)
		}
		if len(p.line)+len(p.data) > maxBody {
			return errors.New("SSE event exceeds limit")
		}
	}
	return nil
}

// These are content-event timing measurements, NOT per-token timings.
type Measurements struct {
	mu        sync.Mutex
	Requests  int       `json:"requests"`
	Failed    int       `json:"failed"`
	Events    int       `json:"content_events"`
	TTFT      []float64 `json:"ttft_content_ms"`
	Intervals []float64 `json:"inter_content_event_ms"`
}

func boundedAppend(xs []float64, x float64) []float64 {
	if len(xs) >= 4096 {
		copy(xs, xs[1:])
		xs = xs[:4095]
	}
	return append(xs, x)
}

func contentEvent(data []byte) bool {
	var event struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(data, &event) != nil {
		return false
	}
	for _, c := range event.Choices {
		if c.Delta.Content != "" {
			return true
		}
	}
	return false
}

type Proxy struct {
	Upstream *url.URL
	Token    string
	APIKey   string
	Client   *http.Client
	Slots    chan struct{}
	Metrics  Measurements
	Timeout  time.Duration
}

func NewProxy(upstream, token string, concurrency int) (*Proxy, error) {
	u, err := url.Parse(upstream)
	if err != nil || u == nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.Fragment != "" {
		return nil, errors.New("UPSTREAM_URL must be a fixed http(s) endpoint without credentials or fragment")
	}
	if token == "" || concurrency < 1 {
		return nil, errors.New("nonempty proxy bearer token and positive concurrency required")
	}
	transport := &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ResponseHeaderTimeout: 10 * time.Second,
		IdleConnTimeout:       30 * time.Second,
		MaxIdleConns:          concurrency,
	}
	return &Proxy{Upstream: u, Token: token, Slots: make(chan struct{}, concurrency), Timeout: 2 * time.Minute,
		Client: &http.Client{Transport: transport, CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		}}}, nil
}

func (p *Proxy) authorized(r *http.Request) bool {
	return subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+p.Token)) == 1
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !p.authorized(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	if r.URL.Path == "/metrics" && r.Method == http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		p.Metrics.mu.Lock()
		defer p.Metrics.mu.Unlock()
		_ = json.NewEncoder(w).Encode(&p.Metrics)
		return
	}
	if r.URL.Path != "/stream" || r.Method != http.MethodPost {
		http.Error(w, "use POST /stream or GET /metrics", http.StatusNotFound)
		return
	}
	select {
	case p.Slots <- struct{}{}:
		defer func() { <-p.Slots }()
	default:
		http.Error(w, "concurrency limit", http.StatusTooManyRequests)
		return
	}
	start := time.Now()
	failed := true
	defer func() {
		p.Metrics.mu.Lock()
		defer p.Metrics.mu.Unlock()
		p.Metrics.Requests++
		if failed {
			p.Metrics.Failed++
		}
	}()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		http.Error(w, "request body exceeds limit", http.StatusRequestEntityTooLarge)
		return
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal(body, &payload) != nil || payload == nil {
		http.Error(w, "JSON object required", http.StatusBadRequest)
		return
	}
	payload["stream"] = json.RawMessage("true")
	body, _ = json.Marshal(payload)
	ctx, cancel := context.WithTimeout(r.Context(), p.Timeout)
	defer cancel()
	upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, p.Upstream.String(), bytes.NewReader(body))
	if err != nil {
		http.Error(w, "upstream configuration error", http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("Accept", "text/event-stream")
	if p.APIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	response, err := p.Client.Do(upstream)
	if err != nil {
		http.Error(w, "upstream unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 ||
		!strings.HasPrefix(strings.ToLower(response.Header.Get("Content-Type")), "text/event-stream") {
		http.Error(w, "upstream did not return a successful SSE response", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("X-Accel-Buffering", "no")
	control := http.NewResponseController(w)
	var previous time.Time
	parser := Parser{OnEvent: func(data []byte) {
		if !contentEvent(data) {
			return
		}
		now := time.Now()
		p.Metrics.mu.Lock()
		defer p.Metrics.mu.Unlock()
		p.Metrics.Events++
		if previous.IsZero() {
			p.Metrics.TTFT = boundedAppend(p.Metrics.TTFT, float64(now.Sub(start))/float64(time.Millisecond))
		} else {
			p.Metrics.Intervals = boundedAppend(p.Metrics.Intervals, float64(now.Sub(previous))/float64(time.Millisecond))
		}
		previous = now
	}}
	buffer, total := make([]byte, 4096), 0
	for {
		n, readErr := response.Body.Read(buffer)
		if n > 0 {
			total += n
			if total > maxStream || parser.Feed(buffer[:n]) != nil {
				_, _ = io.WriteString(w, "\nevent: error\ndata: {\"error\":\"stream limit exceeded\"}\n\n")
				_ = control.Flush()
				return
			}
			if _, err := w.Write(buffer[:n]); err != nil {
				return
			}
			if control.Flush() != nil {
				return
			}
		}
		if readErr != nil {
			failed = !errors.Is(readErr, io.EOF)
			if failed {
				_, _ = io.WriteString(w, "\nevent: error\ndata: {\"error\":\"upstream interrupted\"}\n\n")
				_ = control.Flush()
			}
			return
		}
	}
}

func demo() {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, word := range []string{"Hello", " from", " the", " proxy"} {
			event := map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": word}}}}
			data, _ := json.Marshal(event)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
			w.(http.Flusher).Flush()
			time.Sleep(time.Millisecond)
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	proxy, _ := NewProxy(upstream.URL, "demo-local-only", 1)
	request := httptest.NewRequest("POST", "/stream", strings.NewReader(`{"model":"offline-fixture"}`))
	request.Header.Set("Authorization", "Bearer demo-local-only")
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"http_status": response.Code, "sse": response.Body.String(), "metrics": &proxy.Metrics,
		"timing_unit": "content event; one event may contain multiple tokens",
		"provider":    "local HTTP fixture",
	})
}

func main() {
	showDemo := flag.Bool("demo", false, "run an offline HTTP/SSE demonstration")
	flag.Parse()
	if *showDemo {
		demo()
		return
	}
	proxy, err := NewProxy(os.Getenv("UPSTREAM_URL"), os.Getenv("PROXY_TOKEN"), 8)
	if err != nil {
		log.Fatal(err)
	}
	proxy.APIKey = os.Getenv("UPSTREAM_API_KEY")
	server := &http.Server{Addr: "127.0.0.1:8088", Handler: proxy, ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 15 * time.Second, WriteTimeout: 130 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16384}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()
	fmt.Println("stream proxy listening on 127.0.0.1:8088; bearer authentication required")
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

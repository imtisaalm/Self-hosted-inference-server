package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestParserBoundaries(t *testing.T) {
	for _, newline := range []string{"\n", "\r\n", "\r"} {
		var events []string
		p := Parser{OnEvent: func(b []byte) { events = append(events, string(b)) }}
		input := "\ufeff:comment" + newline + "data: hé" + newline + "data: 世界" + newline + newline + "data: incomplete"
		for _, b := range []byte(input) {
			if err := p.Feed([]byte{b}); err != nil {
				t.Fatal(err)
			}
		}
		if len(events) != 1 || events[0] != "hé\n世界" {
			t.Fatalf("newline %q: %#v", newline, events)
		}
	}
}

func TestParserLimitAndEmptyData(t *testing.T) {
	count := 0
	p := Parser{OnEvent: func(b []byte) { count++ }}
	_ = p.Feed([]byte("data\n\n"))
	if count != 1 {
		t.Fatal(count)
	}
	if p.Feed(bytes.Repeat([]byte{'x'}, maxBody+1)) == nil {
		t.Fatal("oversized event accepted")
	}
}

func TestContentEvent(t *testing.T) {
	for _, c := range []struct {
		data string
		want bool
	}{
		{`{"choices":[{"delta":{"content":"hi"}}]}`, true},
		{`{"choices":[{"delta":{"role":"assistant"}}]}`, false},
		{`{"usage":{"completion_tokens":2}}`, false},
		{`[DONE]`, false}, {`not-json`, false},
	} {
		if contentEvent([]byte(c.data)) != c.want {
			t.Fatal(c.data)
		}
	}
}

func TestForwardAndMetrics(t *testing.T) {
	stream := ": heartbeat\n\ndata: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: [DONE]\n\n"
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer upstream-key" {
			t.Error("credential forwarding")
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["stream"] != true {
			t.Error("stream flag")
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = io.WriteString(w, stream)
	}))
	defer up.Close()
	p, _ := NewProxy(up.URL, "client-key", 2)
	p.APIKey = "upstream-key"
	r := httptest.NewRequest("POST", "/stream", strings.NewReader(`{"model":"fixture"}`))
	r.Header.Set("Authorization", "Bearer client-key")
	w := httptest.NewRecorder()
	p.ServeHTTP(w, r)
	if w.Code != 200 || w.Body.String() != stream || !w.Flushed {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if p.Metrics.Events != 2 || len(p.Metrics.TTFT) != 1 || len(p.Metrics.Intervals) != 1 || p.Metrics.Failed != 0 {
		t.Fatalf("incorrect metrics: events=%d", p.Metrics.Events)
	}
}

func TestAuthValidationCapacity(t *testing.T) {
	p, _ := NewProxy("http://127.0.0.1:9", "key", 1)
	for _, test := range []struct {
		method, path, body, token string
		status                    int
	}{
		{"POST", "/stream", `{}`, "", 401},
		{"GET", "/bad", ``, "key", 404},
		{"POST", "/stream", `[]`, "key", 400},
		{"POST", "/stream", strings.Repeat("x", maxBody+1), "key", 413},
		{"GET", "/metrics", ``, "key", 200},
	} {
		r := httptest.NewRequest(test.method, test.path, strings.NewReader(test.body))
		r.Header.Set("Authorization", "Bearer "+test.token)
		w := httptest.NewRecorder()
		p.ServeHTTP(w, r)
		if w.Code != test.status {
			t.Fatalf("%s got %d want %d", test.path, w.Code, test.status)
		}
	}
	p.Slots <- struct{}{}
	r := httptest.NewRequest("POST", "/stream", strings.NewReader(`{}`))
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	p.ServeHTTP(w, r)
	if w.Code != 429 {
		t.Fatal(w.Code)
	}
}

func TestCancellationReleasesSlot(t *testing.T) {
	cancelled := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(cancelled)
	}))
	defer up.Close()
	p, _ := NewProxy(up.URL, "key", 1)
	p.Timeout = 30 * time.Millisecond
	r := httptest.NewRequest("POST", "/stream", strings.NewReader(`{}`)).WithContext(context.Background())
	r.Header.Set("Authorization", "Bearer key")
	p.ServeHTTP(httptest.NewRecorder(), r)
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("upstream not cancelled")
	}
	if len(p.Slots) != 0 || p.Metrics.Failed != 1 {
		t.Fatal("resources not released")
	}
}

func TestNoRedirectOrNonSSE(t *testing.T) {
	for _, status := range []int{302, 500, 200} {
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Location", "http://127.0.0.1:9")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
		}))
		p, _ := NewProxy(up.URL, "key", 1)
		r := httptest.NewRequest("POST", "/stream", strings.NewReader(`{}`))
		r.Header.Set("Authorization", "Bearer key")
		w := httptest.NewRecorder()
		p.ServeHTTP(w, r)
		up.Close()
		if w.Code != 502 {
			t.Fatal(w.Code)
		}
	}
}

func TestConfiguration(t *testing.T) {
	for _, value := range []string{"file:///etc/passwd", "http://u:p@host/", "http://", ":bad", "http://host/#fragment"} {
		if _, err := NewProxy(value, "key", 1); err == nil {
			t.Fatal(value)
		}
	}
	if _, err := NewProxy("http://localhost", "", 1); err == nil {
		t.Fatal("empty token")
	}
}

func TestFlushBeforeUpstreamEOF(t *testing.T) {
	release := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer up.Close()
	p, _ := NewProxy(up.URL, "key", 1)
	proxy := httptest.NewServer(p)
	defer proxy.Close()
	req, _ := http.NewRequest("POST", proxy.URL+"/stream", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer key")
	client := http.Client{Timeout: time.Second}
	response, err := client.Do(req)
	if err != nil {
		close(release)
		t.Fatal(err)
	}
	buffer := make([]byte, len("data: first\n\n"))
	_, err = io.ReadFull(response.Body, buffer)
	close(release)
	_ = response.Body.Close()
	if err != nil || string(buffer) != "data: first\n\n" {
		t.Fatalf("buffered stream: %s %v", buffer, err)
	}
}

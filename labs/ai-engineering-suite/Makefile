.PHONY: test check eval

test:
	PYTHONPATH=src python -m pytest -q
	cd go/streamproxy && go test -race ./...

check: test
	python -m compileall -q src tests
	cd go/streamproxy && go vet ./...

eval:
	PYTHONPATH=src python -m aieng.evals --baseline data/eval-baseline.json --output /tmp/aieng-eval.json

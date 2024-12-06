# Windows
go build -buildmode=c-shared -o ../python/easyremote/core/easyremote_windows_amd64.dll ./cmd/build/main.go

# Linux
go build -buildmode=c-shared -o ../python/easyremote/core/easyremote_linux_amd64.so ./cmd/build/main.go

# MacOS
go build -buildmode=c-shared -o ../python/easyremote/core/easyremote_darwin_amd64.dylib ./cmd/build/main.go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "runtime"
)

func main() {
    outputDir := "../python/easyremote/core"
    if err := os.MkdirAll(outputDir, 0755); err != nil {
        fmt.Printf("Failed to create output directory: %v\n", err)
        os.Exit(1)
    }

    // 确定输出文件名
    outputName := fmt.Sprintf("easyremote_%s_%s", runtime.GOOS, runtime.GOARCH)
    switch runtime.GOOS {
    case "windows":
        outputName += ".dll"
    case "darwin":
        outputName += ".dylib"
    default:
        outputName += ".so"
    }

    buildOpts := []string{
        "build",
        "-buildmode=c-shared",
        "-o", filepath.Join(outputDir, outputName),
        "./pkg/bridge",
    }

    cmd := exec.Command("go", buildOpts...)
    cmd.Env = append(os.Environ(), "CGO_ENABLED=1")
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr

    fmt.Printf("Building %s...\n", outputName)
    if err := cmd.Run(); err != nil {
        fmt.Printf("Build failed: %v\n", err)
        os.Exit(1)
    }

    fmt.Printf("Successfully built: %s\n", outputName)
}
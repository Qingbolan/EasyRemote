package main

import (
    "flag"
    "fmt"
    "log"
    "os"
    "os/signal"
    "syscall"

    "easycompute/handler"
	easycompute "easycompute/kitex_gen/easycompute/computeservice"
    "github.com/cloudwego/kitex/pkg/rpcinfo"
    "github.com/cloudwego/kitex/server"
)

func main() {
    // 命令行参数
    var (
        nodeID   = flag.String("id", "", "Compute node ID")
        host     = flag.String("host", "0.0.0.0", "Listen host")
        port     = flag.Int("port", 9999, "Listen port")
        vpsAddr  = flag.String("vps", "localhost:8888", "VPS server address")
        tags     = flag.String("tags", "", "Node capabilities tags (comma separated)")
    )
    flag.Parse()

    // 验证必需参数
    if *nodeID == "" {
        log.Fatal("Node ID is required")
    }

    // 创建监听地址
    addr := fmt.Sprintf("%s:%d", *host, *port)

    // 创建handler
    h := handler.NewComputeNodeHandler(
        *nodeID,
        *vpsAddr,
    )

    // 创建服务器选项
    opts := []server.Option{
        server.WithServiceAddr(addr),
        server.WithRegistry(h), // 用于服务注册
        server.WithServerBasicInfo(&rpcinfo.EndpointBasicInfo{
            ServiceName: "compute_node",
            Tags: map[string]string{
                "node_id": *nodeID,
                "tags":    *tags,
            },
        }),
    }

    // 创建服务器
    svr := easycompute.NewServer(h, opts...)

    // 处理退出信号
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    go func() {
        <-sigChan
        log.Println("Shutting down compute node...")
        svr.Stop()
    }()

    // 启动服务器
    log.Printf("Compute Node [%s] starting at %s...", *nodeID, addr)
    if err := svr.Run(); err != nil {
        log.Fatalf("Server stopped with error: %v", err)
    }
}
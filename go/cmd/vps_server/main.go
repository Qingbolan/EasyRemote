package main

import (
    "context"
    "flag"
    "fmt"
    "net"
    "os"
    "os/signal"
    "syscall"
    "time"

    "easycompute/handler"
	easycompute "easycompute/kitex_gen/easycompute/computeservice"
    
    "github.com/cloudwego/kitex/pkg/klog"
    "github.com/cloudwego/kitex/pkg/limit"
    "github.com/cloudwego/kitex/pkg/rpcinfo"
    "github.com/cloudwego/kitex/server"
)

// 服务器配置
type ServerConfig struct {
    Host            string
    Port            int
    MaxConnections  int
    MaxQPS          int
    IdleTimeout     time.Duration
    ExitWaitTime    time.Duration
}

func main() {
    // 1. 解析命令行参数
    config := parseFlags()

    // 2. 初始化日志
    initLogger()

    // 3. 创建服务器Handler
    host := handler.NewVPSHandler()

    // 4. 配置服务器选项
    opts := createServerOptions(config)

    // 5. 创建并配置服务器
    addr := fmt.Sprintf("%s:%d", config.Host, config.Port)
    svr := easycompute.NewServer(host, opts...)

    // 6. 启动服务器监控
    startMetrics(svr)

    // 7. 处理优雅退出
    handleGracefulShutdown(svr, config.ExitWaitTime)

    // 8. 启动服务器
    klog.Infof("VPS Server starting at %s...", addr)
    if err := svr.Run(); err != nil {
        klog.Fatalf("Server stopped with error: %v", err)
    }
}

func parseFlags() *ServerConfig {
    config := &ServerConfig{}
    
    flag.StringVar(&config.Host, "host", "0.0.0.0", "Server host")
    flag.IntVar(&config.Port, "port", 8888, "Server port")
    flag.IntVar(&config.MaxConnections, "max-conns", 50000, "Maximum concurrent connections")
    flag.IntVar(&config.MaxQPS, "max-qps", 10000, "Maximum queries per second")
    flag.DurationVar(&config.IdleTimeout, "idle-timeout", 60*time.Second, "Connection idle timeout")
    flag.DurationVar(&config.ExitWaitTime, "exit-wait", 30*time.Second, "Graceful shutdown wait time")
    
    flag.Parse()

    // 验证配置
    if config.Port <= 0 || config.Port > 65535 {
        klog.Fatalf("Invalid port number: %d", config.Port)
    }

    return config
}

func initLogger() {
    klog.SetLevel(klog.LevelInfo)
    klog.SetOutput(os.Stdout)
}

func createServerOptions(config *ServerConfig) []server.Option {
    opts := []server.Option{
        // 1. 基本服务器信息
        server.WithServerBasicInfo(&rpcinfo.EndpointBasicInfo{
            ServiceName: "vps_server",
            Tags: map[string]string{
                "version": "1.0.0",
            },
        }),

        // 2. 监听地址
        server.WithServiceAddr(&net.TCPAddr{
            IP:   net.ParseIP(config.Host),
            Port: config.Port,
        }),

        // 3. 连接限制
        server.WithLimit(&limit.Option{
            MaxConnections: config.MaxConnections,
            MaxQPS:        config.MaxQPS,
        }),

        // 4. 连接配置
        server.WithReadWriteTimeout(time.Second * 60),

        // 6. 退出配置
        server.WithExitWaitTime(config.ExitWaitTime),
    }

    return opts
}

func startMetrics(svr server.Server) {
    // 这里可以添加metrics收集逻辑
    // 例如: Prometheus指标收集等
    go func() {
        for {
            // 收集服务器状态指标
            time.Sleep(time.Second * 10)
        }
    }()
}

func handleGracefulShutdown(svr server.Server, waitTime time.Duration) {
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

    go func() {
        sig := <-sigChan
        klog.Infof("Received signal %v, initiating graceful shutdown...", sig)

        // 1. 停止接收新请求
        ctx, cancel := context.WithTimeout(context.Background(), waitTime)
        defer cancel()

        // 2. 等待已有请求处理完成
        go func() {
            <-ctx.Done()
            if ctx.Err() == context.DeadlineExceeded {
                klog.Warn("Graceful shutdown timed out, forcing exit...")
            }
        }()

        // 3. 优雅关闭服务器
        svr.Stop()
        klog.Info("Server shutdown completed")
        os.Exit(0)
    }()
}
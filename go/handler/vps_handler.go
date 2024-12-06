// go/handler/vps_handler.go

package handler

import (
    "context"
    "fmt"
    "sync"
    "time"

    "github.com/cloudwego/kitex/pkg/klog"
    "github.com/cloudwego/kitex/client"
    easycompute "easycompute/kitex_gen/easycompute/computeservice"
)

// ComputeNode 表示一个计算节点
type ComputeNode struct {
    ID        string
    Address   string
    Client    easycompute.Client  // RPC客户端
    LastSeen  time.Time          // 最后一次心跳时间
    Stats     NodeStats          // 节点统计
}

// NodeStats 记录节点统计信息
type NodeStats struct {
    ActiveRequests   int64     // 当前活跃请求数
    TotalRequests    int64     // 总请求数
    FailedRequests   int64     // 失败请求数
    LastError        time.Time  // 最后一次错误时间
}

// VPSHandler 处理所有VPS服务器的逻辑
type VPSHandler struct {
    nodes     sync.Map               // 存储所有计算节点，key为节点ID
    stats     map[string]*NodeStats  // 节点统计信息
    statsMu   sync.RWMutex          // 保护统计信息的互斥锁
}

// NewVPSHandler 创建新的VPS处理器
func NewVPSHandler() *VPSHandler {
    h := &VPSHandler{
        stats: make(map[string]*NodeStats),
    }
    
    // 启动节点健康检查
    go h.startHealthCheck()
    
    // 启动统计信息清理
    go h.startStatsCleanup()
    
    return h
}

// RegisterNode 注册新的计算节点
func (h *VPSHandler) RegisterNode(ctx context.Context, nodeID, address string) error {
    // 创建到计算节点的RPC客户端
    cli, err := h.createNodeClient(address)
    if err != nil {
        return fmt.Errorf("failed to create client: %v", err)
    }

    // 创建节点实例
    node := &ComputeNode{
        ID:       nodeID,
        Address:  address,
        Client:   cli,
        LastSeen: time.Now(),
    }

    // 存储节点信息
    h.nodes.Store(nodeID, node)
    
    // 初始化统计信息
    h.statsMu.Lock()
    h.stats[nodeID] = &NodeStats{}
    h.statsMu.Unlock()

    klog.Infof("Node registered: %s at %s", nodeID, address)
    return nil
}

// CallFunction 调用远程函数
func (h *VPSHandler) CallFunction(ctx context.Context, req *easycompute.FunctionCallRequest) (*easycompute.FunctionCallResponse, error) {
    // 获取合适的计算节点
    node, err := h.selectNode(req.TargetNodeId)
    if err != nil {
        return nil, err
    }

    // 更新统计信息
    h.updateStats(node.ID, func(stats *NodeStats) {
        stats.ActiveRequests++
        stats.TotalRequests++
    })
    defer h.updateStats(node.ID, func(stats *NodeStats) {
        stats.ActiveRequests--
    })

    // 调用远程函数
    resp, err := node.Client.CallFunction(ctx, req)
    if err != nil {
        h.updateStats(node.ID, func(stats *NodeStats) {
            stats.FailedRequests++
            stats.LastError = time.Now()
        })
        return nil, fmt.Errorf("call to node %s failed: %v", node.ID, err)
    }

    return resp, nil
}

// StreamFunction 处理流式调用
func (h *VPSHandler) StreamFunction(req *easycompute.FunctionCallRequest, stream easycompute.EasyComputeService_StreamFunctionServer) error {
    // 获取计算节点
    node, err := h.selectNode(req.TargetNodeId)
    if err != nil {
        return err
    }

    // 更新统计信息
    h.updateStats(node.ID, func(stats *NodeStats) {
        stats.ActiveRequests++
        stats.TotalRequests++
    })
    defer h.updateStats(node.ID, func(stats *NodeStats) {
        stats.ActiveRequests--
    })

    // 创建到计算节点的流
    nodeStream, err := node.Client.StreamFunction(context.Background(), req)
    if err != nil {
        h.updateStats(node.ID, func(stats *NodeStats) {
            stats.FailedRequests++
            stats.LastError = time.Now()
        })
        return err
    }

    // 转发流数据
    for {
        resp, err := nodeStream.Recv()
        if err != nil {
            return err
        }

        if err := stream.Send(resp); err != nil {
            return err
        }

        if !resp.HasMore {
            break
        }
    }

    return nil
}

// 选择合适的计算节点
func (h *VPSHandler) selectNode(targetNodeID string) (*ComputeNode, error) {
    if targetNodeID != "" {
        // 如果指定了目标节点
        if node, ok := h.nodes.Load(targetNodeID); ok {
            return node.(*ComputeNode), nil
        }
        return nil, fmt.Errorf("node not found: %s", targetNodeID)
    }

    // 负载均衡选择
    var selectedNode *ComputeNode
    var minLoad int64 = 1<<63 - 1

    h.nodes.Range(func(key, value interface{}) bool {
        node := value.(*ComputeNode)
        
        // 获取节点负载
        h.statsMu.RLock()
        stats := h.stats[node.ID]
        load := stats.ActiveRequests
        h.statsMu.RUnlock()

        if load < minLoad {
            minLoad = load
            selectedNode = node
        }
        return true
    })

    if selectedNode == nil {
        return nil, fmt.Errorf("no available nodes")
    }

    return selectedNode, nil
}

// 创建到计算节点的RPC客户端
func (h *VPSHandler) createNodeClient(address string) (easycompute.Client, error) {
    return easycompute.NewClient(
        "compute_node",
        client.WithHostPorts(address),
        client.WithRPCTimeout(time.Second*30),
        client.WithConnectTimeout(time.Second*3),
        client.WithFailureRetry(retry.NewFailurePolicy()),
    )
}

// 更新节点统计信息
func (h *VPSHandler) updateStats(nodeID string, updater func(*NodeStats)) {
    h.statsMu.Lock()
    defer h.statsMu.Unlock()

    if stats, exists := h.stats[nodeID]; exists {
        updater(stats)
    }
}

// 启动节点健康检查
func (h *VPSHandler) startHealthCheck() {
    ticker := time.NewTicker(time.Second * 10)
    defer ticker.Stop()

    for range ticker.C {
        now := time.Now()
        h.nodes.Range(func(key, value interface{}) bool {
            node := value.(*ComputeNode)
            
            // 检查最后心跳时间
            if now.Sub(node.LastSeen) > time.Second*30 {
                // 节点可能已死，移除它
                h.nodes.Delete(key)
                h.statsMu.Lock()
                delete(h.stats, node.ID)
                h.statsMu.Unlock()
                
                klog.Warnf("Node %s removed due to inactivity", node.ID)
            }
            return true
        })
    }
}

// 清理过期的统计信息
func (h *VPSHandler) startStatsCleanup() {
    ticker := time.NewTicker(time.Hour)
    defer ticker.Stop()

    for range ticker.C {
        h.statsMu.Lock()
        // 清理不存在节点的统计信息
        for nodeID := range h.stats {
            if _, exists := h.nodes.Load(nodeID); !exists {
                delete(h.stats, nodeID)
            }
        }
        h.statsMu.Unlock()
    }
}
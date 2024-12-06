package handler

import (
    "context"

    "sync"
    "time"
    
	easycompute "easycompute/kitex_gen/easycompute/computeservice"
)

type ComputeNodeHandler struct {
    nodeID      string
    pythonProc  *PythonProcess
    funcLock    sync.RWMutex
    vpsClient   easycompute.Client
}

func (h *ComputeNodeHandler) CallFunction(ctx context.Context, req *easycompute.FunctionCallRequest) (*easycompute.FunctionCallResponse, error) {
    // 设置调用超时
    timeoutCtx, cancel := context.WithTimeout(ctx, time.Duration(req.TimeoutMs)*time.Millisecond)
    defer cancel()
    
    // 准备调用参数
    callReq := &PythonCallRequest{
        FuncName: req.FunctionName,
        Args:     req.SerializedArgs,
        Kwargs:   req.SerializedKwargs,
    }
    
    // 调用Python函数
    result, err := h.pythonProc.Call(timeoutCtx, callReq)
    if err != nil {
        return &easycompute.FunctionCallResponse{
            Error: err.Error(),
            NodeId: h.nodeID,
        }, nil
    }
    
    return &easycompute.FunctionCallResponse{
        SerializedResult: result,
        NodeId:          h.nodeID,
    }, nil
}

func (h *ComputeNodeHandler) StreamFunction(req *easycompute.FunctionCallRequest, stream easycompute.EasyComputeService_StreamFunctionServer) error {
    // 创建Python生成器调用
    gen, err := h.pythonProc.CreateGenerator(req.FunctionName, req.SerializedArgs, req.SerializedKwargs)
    if err != nil {
        return err
    }
    defer gen.Close()
    
    // 读取并发送生成器结果
    for {
        result, hasMore, err := gen.Next()
        if err != nil {
            return err
        }
        
        resp := &easycompute.FunctionCallResponse{
            SerializedResult: result,
            HasMore:         hasMore,
            NodeId:          h.nodeID,
        }
        
        if err := stream.Send(resp); err != nil {
            return err
        }
        
        if !hasMore {
            break
        }
    }
    
    return nil
}
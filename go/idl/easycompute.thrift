namespace go easycompute
namespace py easycompute

enum CallType {
    NORMAL = 1
    GENERATOR = 2
    ASYNC = 3
}

// 节点注册请求
struct RegisterRequest {
    1: string node_id
    2: string address
    3: map<string, string> capabilities  // 节点能力
}

struct FunctionCallRequest {
    1: string function_name
    2: binary serialized_args       
    3: binary serialized_kwargs    
    4: CallType call_type         // 调用类型
    5: string target_node_id      // 目标节点ID
    6: i32 timeout_ms             // 超时时间
}

struct FunctionCallResponse {
    1: binary serialized_result  
    2: optional string error
    3: bool has_more            // 用于生成器
    4: string node_id           // 处理节点ID
}

service EasyComputeService {
    // 节点管理
    void RegisterNode(1: RegisterRequest req)
    void Heartbeat(1: string node_id)
    
    // 函数调用
    FunctionCallResponse CallFunction(1: FunctionCallRequest request)
    stream<FunctionCallResponse> StreamFunction(1: FunctionCallRequest request)
}
# MCP Integration Guide

## 🔗 Model Context Protocol Integration

EasyRemote provides seamless integration with the Model Context Protocol (MCP), enabling AI agents and applications to leverage distributed computing resources through a standardized interface.

## 📖 MCP Overview

The Model Context Protocol (MCP) is an open protocol that enables AI models and applications to access external tools and data sources in a secure, standardized way. EasyRemote extends MCP capabilities by providing distributed execution for compute-intensive tools.

### Key Benefits of MCP Integration

- **Standardized Interface**: Use familiar MCP tools with distributed execution
- **Scalable AI Agents**: Build agents that can leverage unlimited compute resources
- **Protocol Compatibility**: Works with any MCP-compatible AI framework
- **Security**: Maintains MCP's security model while adding distributed capabilities

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│  EasyRemote     │◄──►│ ComputeNode     │
│  (AI Agent)     │    │  VPS Gateway    │    │ (Tool Provider) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Standard MCP    │    │ Tool Registry   │    │ Resource Pool   │
│ Protocol        │    │ & API Gateway   │    │ Management      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Implementation Guide

### 1. Setting Up MCP-Compatible ComputeNode

```python
from easyremote import ComputeNode
from easyremote.mcp import mcp_compatible

# Create ComputeNode with MCP-compatible tools
mcp_tools_node = ComputeNode(
    vps_address="ai-gateway.example.com:8080",
    node_id="mcp-ai-tools"
)

@mcp_tools_node.register
@mcp_compatible(
    name="distributed_llm_inference",
    description="Large language model inference across distributed nodes",
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Input prompt"},
            "model": {"type": "string", "default": "llama2-70b"},
            "max_tokens": {"type": "integer", "default": 1000},
            "temperature": {"type": "number", "default": 0.7}
        },
        "required": ["prompt"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "tokens_generated": {"type": "integer"},
            "model_used": {"type": "string"},
            "execution_time": {"type": "number"}
        }
    }
)
def distributed_llm_inference(prompt, model="llama2-70b", max_tokens=1000, temperature=0.7):
    """Execute LLM inference with automatic load balancing"""
    # Find optimal nodes for the model
    optimal_nodes = find_optimal_llm_nodes(model)
    
    # Execute with load balancing
    result = execute_on_least_loaded_node(optimal_nodes, {
        "prompt": prompt,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature
    })
    
    return {
        "text": result.generated_text,
        "tokens_generated": result.token_count,
        "model_used": model,
        "execution_time": result.time_seconds
    }

@mcp_tools_node.register
@mcp_compatible(
    name="parallel_data_processing",
    description="Process large datasets in parallel across multiple nodes",
    input_schema={
        "type": "object",
        "properties": {
            "dataset_path": {"type": "string"},
            "processing_type": {"type": "string", "enum": ["statistical", "ml", "custom"]},
            "chunk_size": {"type": "integer", "default": 1000},
            "output_format": {"type": "string", "default": "json"}
        },
        "required": ["dataset_path", "processing_type"]
    }
)
def parallel_data_processing(dataset_path, processing_type, chunk_size=1000, output_format="json"):
    """Process large datasets in parallel"""
    # Split dataset into chunks
    chunks = split_dataset(dataset_path, chunk_size)
    
    # Find suitable compute nodes
    compute_nodes = find_compute_nodes(
        requirements={"capability": "data_processing", "memory": ">= 8GB"}
    )
    
    # Process chunks in parallel
    results = execute_parallel_tasks(chunks, compute_nodes, processing_type)
    
    # Aggregate results
    aggregated_result = aggregate_results(results)
    
    return {
        "processed_data": aggregated_result,
        "chunks_processed": len(chunks),
        "nodes_used": len(compute_nodes),
        "processing_time": sum(r.time for r in results),
        "output_format": output_format
    }

# Start the MCP-compatible node
mcp_tools_node.serve()
```

### 2. Gateway Configuration for MCP Support

The EasyRemote VPS Gateway automatically generates MCP-compatible endpoints for registered functions.

```python
from easyremote import Server
from easyremote.mcp import MCPGateway

# Enhanced server with MCP support
class EasyRemoteMCPServer(Server):
    def __init__(self, port=8080):
        super().__init__(port)
        self.mcp_gateway = MCPGateway(self)
        
        # Automatically generate MCP endpoints
        self.setup_mcp_routes()
    
    def setup_mcp_routes(self):
        """Setup MCP protocol endpoints"""
        
        @self.app.get("/mcp/tools")
        async def list_mcp_tools():
            """List all available MCP-compatible tools"""
            mcp_tools = []
            
            for function_id, function_info in self.function_registry.items():
                if function_info.get('mcp_compatible', False):
                    mcp_tools.append({
                        "name": function_info["mcp_name"],
                        "description": function_info["description"],
                        "inputSchema": function_info["input_schema"],
                        "outputSchema": function_info.get("output_schema", {}),
                        "provider": function_info["node_id"]
                    })
            
            return {"tools": mcp_tools}
        
        @self.app.post("/mcp/tools/{tool_name}")
        async def execute_mcp_tool(tool_name: str, request: dict):
            """Execute MCP tool with distributed execution"""
            # Find the actual function
            function_id = self.mcp_gateway.get_function_id(tool_name)
            
            # Route to appropriate compute node
            result = await self.route_to_node(function_id, request)
            
            return {"result": result}
        
        @self.app.post("/mcp/resources")
        async def get_mcp_resources():
            """Get available MCP resources"""
            return {
                "resources": [
                    {
                        "uri": f"easyremote://{node_id}",
                        "name": f"EasyRemote Node {node_id}",
                        "description": f"Distributed compute resources on {node_id}",
                        "mimeType": "application/json"
                    }
                    for node_id in self.get_connected_nodes()
                ]
            }

# Start server with MCP support
server = EasyRemoteMCPServer(port=8080)
server.run()
```

### 3. Creating MCP-Compatible AI Agent

```python
from easyremote import remote
from easyremote.mcp import MCPClient
import asyncio

class DistributedAIAgent:
    def __init__(self, mcp_server_url="http://localhost:8080"):
        self.mcp_server_url = mcp_server_url
        self.mcp_client = MCPClient(mcp_server_url)
        
        # Also can use direct EasyRemote integration
        self.llm_inference = remote("mcp-ai-tools", "distributed_llm_inference")
        self.data_processing = remote("mcp-ai-tools", "parallel_data_processing")
        
        # Connect to MCP server
        self.setup_mcp_connection()
    
    async def setup_mcp_connection(self):
        """Establish connection to MCP server"""
        await self.mcp_client.connect()
        
        # Discover available tools
        self.available_tools = await self.mcp_client.list_tools()
        print(f"Connected to MCP server with {len(self.available_tools)} tools")
    
    async def process_user_request(self, request: str):
        """Process user request using distributed MCP tools"""
        # Method 1: Use MCP protocol
        analysis_result = await self.mcp_client.call_tool(
            "distributed_llm_inference",
            {
                "prompt": f"Analyze this request and determine what tools are needed: {request}",
                "model": "gpt-4",
                "max_tokens": 500
            }
        )
        
        # Method 2: Use direct EasyRemote integration
        tool_plan = await self.llm_inference(
            prompt=f"Create a step-by-step plan for: {request}",
            model="gpt-4",
            max_tokens=800
        )
        
        # Execute the plan
        results = await self.execute_tool_plan(tool_plan["text"])
        
        # Synthesize final response
        final_response = await self.llm_inference(
            prompt=f"""
            Original request: {request}
            Execution results: {results}
            
            Please synthesize these results into a coherent, helpful response.
            """,
            model="gpt-4",
            max_tokens=1000
        )
        
        return {
            "response": final_response["text"],
            "tools_used": list(results.keys()),
            "execution_time": sum(r.get("execution_time", 0) for r in results.values()),
            "protocol": "mcp+easyremote"
        }
    
    async def execute_tool_plan(self, plan_text):
        """Execute a multi-step tool plan"""
        # Parse plan into actionable steps
        steps = self.parse_plan(plan_text)
        
        results = {}
        for step in steps:
            if step["tool"] == "data_processing":
                result = await self.data_processing(
                    dataset_path=step["dataset"],
                    processing_type=step["type"]
                )
                results[step["tool"]] = result
            
            elif step["tool"] == "llm_inference":
                result = await self.llm_inference(
                    prompt=step["prompt"],
                    model=step.get("model", "llama2-70b")
                )
                results[step["tool"]] = result
        
        return results

# Usage example
agent = DistributedAIAgent()

# Process complex request that requires distributed computing
response = await agent.process_user_request(
    "Analyze the sales data from Q3 2024 and generate insights with recommendations"
)
```

## 🔧 Advanced MCP Features

### Custom Tool Development

```python
from easyremote import ComputeNode
from easyremote.mcp import mcp_tool, MCPToolMetadata

class CustomMCPToolNode(ComputeNode):
    def __init__(self, node_id: str, vps_address: str):
        super().__init__(vps_address, node_id)
        self.register_custom_tools()
    
    def register_custom_tools(self):
        """Register custom MCP tools"""
        
        @self.register
        @mcp_tool(
            metadata=MCPToolMetadata(
                name="distributed_image_generation",
                description="Generate images in parallel using distributed GPU resources",
                category="image_generation",
                tags=["ai", "gpu", "parallel"]
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of image generation prompts"
                    },
                    "style": {"type": "string", "default": "realistic"},
                    "resolution": {"type": "string", "default": "1024x1024"},
                    "num_images_per_prompt": {"type": "integer", "default": 1}
                },
                "required": ["prompts"]
            }
        )
        def distributed_image_generation(self, prompts, style="realistic", resolution="1024x1024", num_images_per_prompt=1):
            """Generate images using distributed GPU resources"""
            
            # Find available GPU nodes
            gpu_nodes = self.find_gpu_nodes(
                requirements={"capability": "image_generation", "memory": ">= 12GB"}
            )
            
            # Distribute prompts across GPU nodes
            task_distribution = self.distribute_image_tasks(
                prompts=prompts,
                nodes=gpu_nodes,
                images_per_prompt=num_images_per_prompt
            )
            
            # Execute image generation in parallel
            image_results = asyncio.gather(*[
                self.generate_images_on_node(node, tasks, style, resolution)
                for node, tasks in task_distribution.items()
            ])
            
            # Collect and format results
            all_images = []
            for node_results in image_results:
                all_images.extend(node_results)
            
            return {
                "images": all_images,
                "metadata": {
                    "total_images": len(all_images),
                    "prompts_processed": len(prompts),
                    "nodes_used": len(gpu_nodes),
                    "style": style,
                    "resolution": resolution
                }
            }
        
        @self.register
        @mcp_tool(
            metadata=MCPToolMetadata(
                name="distributed_code_execution",
                description="Execute code across distributed compute resources",
                category="computation",
                security_level="high"
            )
        )
        def distributed_code_execution(self, code, language="python", requirements=None):
            """Execute code on distributed compute nodes with security"""
            
            # Security validation
            if not self.validate_code_security(code, language):
                raise SecurityError("Code failed security validation")
            
            # Find suitable execution nodes
            exec_nodes = self.find_execution_nodes(
                language=language,
                requirements=requirements or {}
            )
            
            # Execute with sandboxing
            result = self.execute_sandboxed_code(
                code=code,
                language=language,
                nodes=exec_nodes
            )
            
            return {
                "output": result.stdout,
                "errors": result.stderr,
                "execution_time": result.time,
                "exit_code": result.exit_code,
                "node_used": result.node_id
            }

# Deploy custom MCP tools
custom_node = CustomMCPToolNode("custom-mcp-tools", "gateway:8080")
custom_node.serve()
```

### MCP Resource Management

```python
from easyremote.mcp import MCPResourceManager

class DistributedMCPResourceManager:
    def __init__(self, gateway_url):
        self.gateway_url = gateway_url
        self.resource_manager = MCPResourceManager()
    
    async def register_distributed_resources(self):
        """Register distributed resources as MCP resources"""
        
        # Register compute nodes as resources
        compute_nodes = await self.get_connected_nodes()
        
        for node in compute_nodes:
            await self.resource_manager.register_resource({
                "uri": f"easyremote://compute/{node.id}",
                "name": f"Compute Node {node.id}",
                "description": f"Distributed compute resource: {node.capabilities}",
                "mimeType": "application/json",
                "metadata": {
                    "node_type": "compute",
                    "capabilities": node.capabilities,
                    "current_load": node.current_load,
                    "availability": node.availability
                }
            })
        
        # Register data sources
        data_sources = await self.get_data_sources()
        
        for source in data_sources:
            await self.resource_manager.register_resource({
                "uri": f"easyremote://data/{source.id}",
                "name": f"Data Source {source.name}",
                "description": f"Distributed data resource: {source.description}",
                "mimeType": source.mime_type,
                "metadata": {
                    "size": source.size,
                    "last_updated": source.last_updated,
                    "access_level": source.access_level
                }
            })
    
    async def get_resource_content(self, uri: str):
        """Get content of a distributed resource"""
        if uri.startswith("easyremote://compute/"):
            node_id = uri.split("/")[-1]
            node_info = await self.get_node_info(node_id)
            return {
                "content": node_info,
                "mimeType": "application/json"
            }
        
        elif uri.startswith("easyremote://data/"):
            data_id = uri.split("/")[-1]
            data_content = await self.fetch_data_content(data_id)
            return {
                "content": data_content,
                "mimeType": "application/json"
            }
        
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

# Setup distributed resource management
resource_manager = DistributedMCPResourceManager("http://gateway:8080")
await resource_manager.register_distributed_resources()
```

## 🎯 Integration Examples

### Claude Desktop Integration

```json
{
  "mcpServers": {
    "easyremote": {
      "command": "python",
      "args": ["-m", "easyremote.mcp.server"],
      "env": {
        "EASYREMOTE_GATEWAY": "http://your-gateway:8080",
        "EASYREMOTE_API_KEY": "your-api-key"
      }
    }
  }
}
```

### VS Code Extension Integration

```typescript
// VS Code extension for EasyRemote MCP integration
import { MCPClient } from '@modelcontextprotocol/client';

class EasyRemoteMCPExtension {
    private mcpClient: MCPClient;
    
    constructor() {
        this.mcpClient = new MCPClient('http://localhost:8080/mcp');
    }
    
    async executeDistributedCode(code: string, language: string) {
        const result = await this.mcpClient.callTool('distributed_code_execution', {
            code: code,
            language: language,
            requirements: {
                memory: '4GB',
                timeout: 300
            }
        });
        
        return result;
    }
    
    async generateAIResponse(prompt: string, model: string = 'llama2-70b') {
        const result = await this.mcpClient.callTool('distributed_llm_inference', {
            prompt: prompt,
            model: model,
            max_tokens: 1000
        });
        
        return result.text;
    }
}
```

### Jupyter Notebook Integration

```python
# Jupyter magic command for EasyRemote MCP
from IPython.core.magic import line_magic, cell_magic, Magics, magics_class
from easyremote.mcp import MCPClient

@magics_class
class EasyRemoteMagics(Magics):
    
    def __init__(self, shell):
        super().__init__(shell)
        self.mcp_client = MCPClient('http://localhost:8080/mcp')
    
    @line_magic
    def easyremote_llm(self, line):
        """Execute LLM inference on distributed nodes"""
        result = await self.mcp_client.call_tool('distributed_llm_inference', {
            'prompt': line,
            'model': 'gpt-4'
        })
        return result['text']
    
    @cell_magic
    def easyremote_compute(self, line, cell):
        """Execute code on distributed compute nodes"""
        args = line.split()
        language = args[0] if args else 'python'
        
        result = await self.mcp_client.call_tool('distributed_code_execution', {
            'code': cell,
            'language': language
        })
        
        print(result['output'])
        if result['errors']:
            print("Errors:", result['errors'])

# Load the extension
ip = get_ipython()
ip.register_magic_function(EasyRemoteMagics)

# Usage in Jupyter
# %easyremote_llm What is quantum computing?
# %%easyremote_compute python
# import numpy as np
# result = np.random.rand(1000000).sum()
# print(f"Result: {result}")
```

## 🔐 Security and Authentication

### MCP Authentication

```python
from easyremote.mcp import MCPAuthenticator

class SecureMCPIntegration:
    def __init__(self):
        self.authenticator = MCPAuthenticator()
        self.auth_config = {
            "require_authentication": True,
            "allowed_origins": ["https://claude.ai", "vscode://"],
            "api_key_required": True
        }
    
    async def authenticate_mcp_request(self, request):
        """Authenticate MCP requests"""
        # Verify API key
        if not await self.authenticator.verify_api_key(request.api_key):
            raise AuthenticationError("Invalid API key")
        
        # Verify origin
        if request.origin not in self.auth_config["allowed_origins"]:
            raise AuthenticationError(f"Origin {request.origin} not allowed")
        
        # Verify tool permissions
        if not await self.authenticator.check_tool_permissions(
            request.user_id, request.tool_name
        ):
            raise AuthorizationError("Insufficient permissions for tool")
        
        return True
    
    async def secure_tool_execution(self, tool_name, params, user_context):
        """Execute tool with security controls"""
        # Validate input parameters
        validated_params = await self.validate_input_parameters(tool_name, params)
        
        # Apply rate limiting
        await self.check_rate_limits(user_context.user_id, tool_name)
        
        # Execute with monitoring
        result = await self.execute_monitored_tool(tool_name, validated_params)
        
        # Audit logging
        await self.log_tool_execution(user_context, tool_name, result)
        
        return result
```

## 📊 Monitoring and Analytics

### MCP Usage Analytics

```python
from easyremote.mcp import MCPAnalytics

class MCPUsageAnalytics:
    def __init__(self):
        self.analytics = MCPAnalytics()
    
    async def track_tool_usage(self, tool_name, execution_time, success, user_id):
        """Track MCP tool usage for analytics"""
        await self.analytics.record_event({
            "event_type": "tool_execution",
            "tool_name": tool_name,
            "execution_time": execution_time,
            "success": success,
            "user_id": user_id,
            "timestamp": time.time()
        })
    
    async def generate_usage_report(self, time_period="24h"):
        """Generate usage analytics report"""
        metrics = await self.analytics.get_metrics(time_period)
        
        return {
            "total_tool_calls": metrics.total_calls,
            "average_execution_time": metrics.avg_execution_time,
            "success_rate": metrics.success_rate,
            "most_used_tools": metrics.top_tools,
            "resource_utilization": metrics.resource_usage,
            "cost_analysis": metrics.cost_breakdown
        }
    
    async def optimize_performance(self):
        """Analyze performance and suggest optimizations"""
        performance_data = await self.analytics.get_performance_data()
        
        optimizations = []
        
        # Identify slow tools
        slow_tools = [tool for tool in performance_data.tools 
                     if tool.avg_execution_time > 30]
        
        for tool in slow_tools:
            optimizations.append({
                "tool": tool.name,
                "issue": "slow_execution",
                "recommendation": "Consider caching or load balancing",
                "potential_improvement": f"{tool.avg_execution_time * 0.6:.1f}s faster"
            })
        
        return optimizations

# Usage
analytics = MCPUsageAnalytics()
report = await analytics.generate_usage_report("7d")
optimizations = await analytics.optimize_performance()
```

## 🚀 Best Practices

### 1. Tool Design
- **Single Responsibility**: Each MCP tool should have a clear, focused purpose
- **Descriptive Schemas**: Provide comprehensive input/output schemas
- **Error Handling**: Implement robust error handling with meaningful messages
- **Resource Awareness**: Tools should be aware of resource requirements

### 2. Performance Optimization
- **Caching**: Implement intelligent caching for frequently called tools
- **Load Balancing**: Distribute tool execution across multiple nodes
- **Async Execution**: Use asynchronous execution for better concurrency
- **Resource Pooling**: Pool expensive resources like GPU memory

### 3. Security Best Practices
- **Input Validation**: Always validate and sanitize inputs
- **Authentication**: Implement proper authentication for all MCP endpoints
- **Rate Limiting**: Prevent abuse through rate limiting
- **Audit Logging**: Log all tool executions for security and compliance

### 4. Integration Guidelines
- **Standard Compliance**: Follow MCP protocol specifications exactly
- **Backward Compatibility**: Maintain compatibility with existing MCP clients
- **Documentation**: Provide comprehensive documentation for all tools
- **Testing**: Implement thorough testing for all MCP integrations

---

*The MCP integration enables EasyRemote to serve as a powerful backend for AI agents while maintaining compatibility with the broader MCP ecosystem and leveraging the simple three-role EasyRemote architecture.* 
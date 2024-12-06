package main

/*
#include <stdlib.h>
*/
import "C"
import (
    "encoding/json"
    "fmt"
    "log"
    "unsafe"
    
    "easyremote/pkg/processor"
    "easyremote/pkg/types"
)

var proc = processor.NewProcessor(true)

//export ProcessData
func ProcessData(cData *C.char, length C.int, cMetadata *C.char) *C.char {
    log.Printf("ProcessData called with data length: %d", length)

    // 转换输入数据
    inputData := C.GoBytes(unsafe.Pointer(cData), length)
    metadataStr := C.GoString(cMetadata)
    
    log.Printf("Received metadata string: %s", metadataStr)
    
    var metadata types.Metadata
    if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
        errMsg := formatError("metadata unmarshal error", err)
        log.Printf("Failed to unmarshal metadata: %v", err)
        return C.CString(errMsg)
    }

    log.Printf("Parsed metadata: %+v", metadata)

    // 创建数据包
    packet := &types.DataPacket{
        Data:     inputData,
        Metadata: metadata,
    }

    // 处理数据
    log.Printf("Processing data packet...")
    result, err := proc.Process(packet)
    if err != nil {
        errMsg := formatError("processing error", err)
        log.Printf("Failed to process data: %v", err)
        return C.CString(errMsg)
    }

    // 序列化结果
    log.Printf("Marshaling result...")
    resultJSON, err := json.Marshal(result)
    if err != nil {
        errMsg := formatError("result marshal error", err)
        log.Printf("Failed to marshal result: %v", err)
        return C.CString(errMsg)
    }

    log.Printf("Successfully processed data, result length: %d", len(resultJSON))
    return C.CString(string(resultJSON))
}

//export FreeString
func FreeString(str *C.char) {
    if str != nil {
        log.Printf("Freeing string memory")
        C.free(unsafe.Pointer(str))
    }
}

// 格式化错误信息为 JSON 格式
func formatError(msg string, err error) string {
    errorStruct := struct {
        Error string `json:"error"`
    }{
        Error: fmt.Sprintf("%s: %v", msg, err),
    }

    jsonBytes, err := json.Marshal(errorStruct)
    if err != nil {
        // 如果连错误序列化都失败了，返回简单的错误字符串
        return fmt.Sprintf(`{"error":"failed to format error message: %v"}`, err)
    }

    return string(jsonBytes)
}

func main() {
    log.Printf("Bridge initialized with compression enabled")
}
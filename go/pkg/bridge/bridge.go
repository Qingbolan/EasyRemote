package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"
import (
	"easyremote/pkg/processor"
	"easyremote/pkg/types"
	// "log"
	"unsafe"
)

var proc = processor.NewProcessor(true)

//export ProcessData
func ProcessData(cData *C.char, length C.size_t, outLength *C.size_t) *C.char {
	// log.Printf("ProcessData called with data length: %d", length)

	// 转换输入数据
	inputData := C.GoBytes(unsafe.Pointer(cData), C.int(length))

	// 创建数据包
	packet := &types.DataPacket{
		Data: inputData,
	}

	// 处理数据
	// log.Printf("Processing data packet...")
	result, err := proc.Process(packet)
	if err != nil {
		// log.Printf("Failed to process data: %v", err)
		return nil
	}

	// 分配内存并返回结果
	totalLength := len(result.Data)
	cResultData := C.CBytes(result.Data)
	*outLength = C.size_t(totalLength)

	// log.Printf("Go: totalLength = %d", totalLength)
	// log.Printf("Go: *outLength = %d", *outLength)

	return (*C.char)(cResultData)
}

//export FreeResult
func FreeResult(ptr *C.char) {
	if ptr != nil {
		C.free(unsafe.Pointer(ptr))
	}
}

func main() {
	// log.Printf("Bridge initialized with compression enabled")
}

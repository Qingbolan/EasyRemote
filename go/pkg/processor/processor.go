package processor

import (
    "bytes"
    "compress/gzip"
    "fmt"
    "io"
    "log"
    "sync"

    "easyremote/pkg/types"
)

// LogLevel defines the logging level for the processor
type LogLevel int

const (
    // DEBUG shows all log messages
    DEBUG LogLevel = iota
    // INFO shows only important info and errors
    INFO
    // ERROR shows only error messages
    ERROR
)

// Processor handles data packet processing with optional compression
type Processor struct {
    useCompression bool           // 是否使用压缩
    bufferPool     sync.Pool      // 用于重用缓冲区的对象池
    logLevel       LogLevel       // 日志级别控制
}

// NewProcessor creates a new Processor instance with specified compression setting
func NewProcessor(useCompression bool) *Processor {
    return &Processor{
        useCompression: useCompression,
        logLevel:       DEBUG,  // 默认使用INFO级别
        bufferPool: sync.Pool{
            New: func() interface{} {
                return new(bytes.Buffer)
            },
        },
    }
}

// SetLogLevel allows changing the logging level
func (p *Processor) SetLogLevel(level LogLevel) {
    p.logLevel = level
}

// logMessage handles logging based on the current log level
func (p *Processor) logMessage(level LogLevel, message string) {
    if level >= p.logLevel {
        log.Printf("[%v] %s", level, message)
    }
}

// Process handles the main data processing pipeline
func (p *Processor) Process(packet *types.DataPacket) (*types.DataPacket, error) {
    p.logMessage(DEBUG, "Starting data processing...")
    
    // 验证元数据
    if err := p.ValidateMetadata(packet.Metadata); err != nil {
        p.logMessage(ERROR, fmt.Sprintf("Metadata validation failed: %v", err))
        return nil, fmt.Errorf("metadata validation failed: %v", err)
    }
    p.logMessage(DEBUG, "Metadata validation passed")

    data := packet.Data
    if p.useCompression {
        p.logMessage(DEBUG, "Decompressing data...")
        var err error
        data, err = p.decompress(data)
        if err != nil {
            p.logMessage(ERROR, fmt.Sprintf("Decompression failed: %v", err))
            return nil, fmt.Errorf("decompress error: %v", err)
        }
        p.logMessage(DEBUG, "Data decompressed successfully")
    }

    // 处理数据(目前只是传递)
    processedData := data
    p.logMessage(INFO, fmt.Sprintf("Processed data size: %d bytes", len(processedData)))

    if p.useCompression {
        p.logMessage(DEBUG, "Compressing processed data...")
        var err error
        processedData, err = p.compress(processedData)
        if err != nil {
            p.logMessage(ERROR, fmt.Sprintf("Compression failed: %v", err))
            return nil, fmt.Errorf("compress error: %v", err)
        }
        p.logMessage(DEBUG, "Data compressed successfully")
    }

    p.logMessage(DEBUG, "Processing completed successfully")
    return &types.DataPacket{
        Data:     processedData,
        Metadata: packet.Metadata,
    }, nil
}

// compress compresses the input data using gzip
func (p *Processor) compress(data []byte) ([]byte, error) {
    p.logMessage(DEBUG, "Starting compression...")
    buf := p.bufferPool.Get().(*bytes.Buffer)
    buf.Reset()
    defer p.bufferPool.Put(buf)

    gz := gzip.NewWriter(buf)
    if _, err := gz.Write(data); err != nil {
        return nil, err
    }
    if err := gz.Close(); err != nil {
        return nil, err
    }

    result := make([]byte, buf.Len())
    copy(result, buf.Bytes())
    p.logMessage(DEBUG, fmt.Sprintf("Compressed size: %d bytes", len(result)))
    return result, nil
}

// decompress decompresses the input data using gzip
func (p *Processor) decompress(data []byte) ([]byte, error) {
    p.logMessage(DEBUG, "Starting decompression...")
    gz, err := gzip.NewReader(bytes.NewReader(data))
    if err != nil {
        return nil, err
    }
    defer gz.Close()

    buf := p.bufferPool.Get().(*bytes.Buffer)
    buf.Reset()
    defer p.bufferPool.Put(buf)

    if _, err := io.Copy(buf, gz); err != nil {
        return nil, err
    }

    result := make([]byte, buf.Len())
    copy(result, buf.Bytes())
    p.logMessage(DEBUG, fmt.Sprintf("Decompressed size: %d bytes", len(result)))
    return result, nil
}

// ValidateMetadata validates the metadata based on its type
func (p *Processor) ValidateMetadata(metadata types.Metadata) error {
    p.logMessage(DEBUG, fmt.Sprintf("Validating metadata of type: %v", metadata.Type))
    
    switch metadata.Type {
    case types.TypeNDArray, types.TypeVideo:
        if len(metadata.Shape) == 0 || metadata.Dtype == "" {
            return fmt.Errorf("invalid array metadata: shape and dtype required")
        }
    case types.TypeImage:
        if metadata.Format == "" || len(metadata.Size) != 2 {
            return fmt.Errorf("invalid image metadata: format and size required")
        }
    case types.TypeAudio:
        if len(metadata.Shape) != 2 || metadata.Channels == 0 {
            return fmt.Errorf("invalid audio metadata: shape and channels required")
        }
    }
    return nil
}
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

type LogLevel int

const (
	DEBUG LogLevel = iota
	INFO
	ERROR
)

type Processor struct {
	useCompression bool
	bufferPool     sync.Pool
	logLevel       LogLevel
}

func NewProcessor(useCompression bool) *Processor {
	return &Processor{
		useCompression: useCompression,
		logLevel:       ERROR,
		bufferPool: sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		},
	}
}

func (p *Processor) SetLogLevel(level LogLevel) {
	p.logLevel = level
}

func (p *Processor) logMessage(level LogLevel, message string) {
	if level >= p.logLevel {
		log.Printf("[%v] %s", level, message)
	}
}

func (p *Processor) Process(packet *types.DataPacket) (*types.DataPacket, error) {
	p.logMessage(DEBUG, "Starting data processing...")

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

	// 处理数据（当前直接返回）
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
		Data: processedData,
	}, nil
}

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

package types

import "encoding/json"

type DataType string

const (
    TypeNull    DataType = "null"
    TypeInt     DataType = "int"
    TypeFloat   DataType = "float"
    TypeBool    DataType = "bool"
    TypeStr     DataType = "str"
    TypeBytes   DataType = "bytes"
    TypeList    DataType = "list"
    TypeDict    DataType = "dict"
    TypeSet     DataType = "set"
    TypeImage   DataType = "image"
    TypeVideo   DataType = "video"
    TypeAudio   DataType = "audio"
    TypeNDArray DataType = "ndarray"
)

type Metadata struct {
    Type     DataType            `json:"type"`
    Shape    []int              `json:"shape,omitempty"`
    Dtype    string             `json:"dtype,omitempty"`
    Format   string             `json:"format,omitempty"`
    Mode     string             `json:"mode,omitempty"`
    Size     []int              `json:"size,omitempty"`
    Channels int                `json:"channels,omitempty"`
    Extra    map[string]string  `json:"extra,omitempty"`
}

type DataPacket struct {
    Data     []byte   `json:"data"`
    Metadata Metadata `json:"metadata"`
}

func (p *DataPacket) ToJSON() ([]byte, error) {
    return json.Marshal(p)
}

func FromJSON(data []byte) (*DataPacket, error) {
    var packet DataPacket
    err := json.Unmarshal(data, &packet)
    return &packet, err
}

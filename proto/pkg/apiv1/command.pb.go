// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// source: determined/api/v1/command.proto

package apiv1

import (
	commandv1 "github.com/determined-ai/determined/proto/pkg/commandv1"
	utilv1 "github.com/determined-ai/determined/proto/pkg/utilv1"
	_struct "github.com/golang/protobuf/ptypes/struct"
	_ "github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger/options"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// Enum values for warnings when launching commands.
type LaunchWarning int32

const (
	// Default value
	LaunchWarning_LAUNCH_WARNING_UNSPECIFIED LaunchWarning = 0
	// For a default webhook
	LaunchWarning_LAUNCH_WARNING_CURRENT_SLOTS_EXCEEDED LaunchWarning = 1
)

// Enum value maps for LaunchWarning.
var (
	LaunchWarning_name = map[int32]string{
		0: "LAUNCH_WARNING_UNSPECIFIED",
		1: "LAUNCH_WARNING_CURRENT_SLOTS_EXCEEDED",
	}
	LaunchWarning_value = map[string]int32{
		"LAUNCH_WARNING_UNSPECIFIED":            0,
		"LAUNCH_WARNING_CURRENT_SLOTS_EXCEEDED": 1,
	}
)

func (x LaunchWarning) Enum() *LaunchWarning {
	p := new(LaunchWarning)
	*p = x
	return p
}

func (x LaunchWarning) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (LaunchWarning) Descriptor() protoreflect.EnumDescriptor {
	return file_determined_api_v1_command_proto_enumTypes[0].Descriptor()
}

func (LaunchWarning) Type() protoreflect.EnumType {
	return &file_determined_api_v1_command_proto_enumTypes[0]
}

func (x LaunchWarning) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use LaunchWarning.Descriptor instead.
func (LaunchWarning) EnumDescriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{0}
}

// Sorts commands by the given field.
type GetCommandsRequest_SortBy int32

const (
	// Returns commands in an unsorted list.
	GetCommandsRequest_SORT_BY_UNSPECIFIED GetCommandsRequest_SortBy = 0
	// Returns commands sorted by id.
	GetCommandsRequest_SORT_BY_ID GetCommandsRequest_SortBy = 1
	// Returns commands sorted by description.
	GetCommandsRequest_SORT_BY_DESCRIPTION GetCommandsRequest_SortBy = 2
	// Return commands sorted by start time.
	GetCommandsRequest_SORT_BY_START_TIME GetCommandsRequest_SortBy = 4
)

// Enum value maps for GetCommandsRequest_SortBy.
var (
	GetCommandsRequest_SortBy_name = map[int32]string{
		0: "SORT_BY_UNSPECIFIED",
		1: "SORT_BY_ID",
		2: "SORT_BY_DESCRIPTION",
		4: "SORT_BY_START_TIME",
	}
	GetCommandsRequest_SortBy_value = map[string]int32{
		"SORT_BY_UNSPECIFIED": 0,
		"SORT_BY_ID":          1,
		"SORT_BY_DESCRIPTION": 2,
		"SORT_BY_START_TIME":  4,
	}
)

func (x GetCommandsRequest_SortBy) Enum() *GetCommandsRequest_SortBy {
	p := new(GetCommandsRequest_SortBy)
	*p = x
	return p
}

func (x GetCommandsRequest_SortBy) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (GetCommandsRequest_SortBy) Descriptor() protoreflect.EnumDescriptor {
	return file_determined_api_v1_command_proto_enumTypes[1].Descriptor()
}

func (GetCommandsRequest_SortBy) Type() protoreflect.EnumType {
	return &file_determined_api_v1_command_proto_enumTypes[1]
}

func (x GetCommandsRequest_SortBy) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use GetCommandsRequest_SortBy.Descriptor instead.
func (GetCommandsRequest_SortBy) EnumDescriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{0, 0}
}

// Get a list of commands.
type GetCommandsRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Sort commands by the given field.
	SortBy GetCommandsRequest_SortBy `protobuf:"varint,1,opt,name=sort_by,json=sortBy,proto3,enum=determined.api.v1.GetCommandsRequest_SortBy" json:"sort_by,omitempty"`
	// Order commands in either ascending or descending order.
	OrderBy OrderBy `protobuf:"varint,2,opt,name=order_by,json=orderBy,proto3,enum=determined.api.v1.OrderBy" json:"order_by,omitempty"`
	// Skip the number of commands before returning results. Negative values
	// denote number of commands to skip from the end before returning results.
	Offset int32 `protobuf:"varint,3,opt,name=offset,proto3" json:"offset,omitempty"`
	// Limit the number of commands. A value of 0 denotes no limit.
	Limit int32 `protobuf:"varint,4,opt,name=limit,proto3" json:"limit,omitempty"`
	// Limit commands to those that are owned by users with the specified
	// usernames.
	Users []string `protobuf:"bytes,5,rep,name=users,proto3" json:"users,omitempty"`
	// Limit commands to those that are owned by users with the specified userIds.
	UserIds []int32 `protobuf:"varint,6,rep,packed,name=user_ids,json=userIds,proto3" json:"user_ids,omitempty"`
}

func (x *GetCommandsRequest) Reset() {
	*x = GetCommandsRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetCommandsRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetCommandsRequest) ProtoMessage() {}

func (x *GetCommandsRequest) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetCommandsRequest.ProtoReflect.Descriptor instead.
func (*GetCommandsRequest) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{0}
}

func (x *GetCommandsRequest) GetSortBy() GetCommandsRequest_SortBy {
	if x != nil {
		return x.SortBy
	}
	return GetCommandsRequest_SORT_BY_UNSPECIFIED
}

func (x *GetCommandsRequest) GetOrderBy() OrderBy {
	if x != nil {
		return x.OrderBy
	}
	return OrderBy_ORDER_BY_UNSPECIFIED
}

func (x *GetCommandsRequest) GetOffset() int32 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *GetCommandsRequest) GetLimit() int32 {
	if x != nil {
		return x.Limit
	}
	return 0
}

func (x *GetCommandsRequest) GetUsers() []string {
	if x != nil {
		return x.Users
	}
	return nil
}

func (x *GetCommandsRequest) GetUserIds() []int32 {
	if x != nil {
		return x.UserIds
	}
	return nil
}

// Response to GetCommandsRequest.
type GetCommandsResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The list of returned commands.
	Commands []*commandv1.Command `protobuf:"bytes,1,rep,name=commands,proto3" json:"commands,omitempty"`
	// Pagination information of the full dataset.
	Pagination *Pagination `protobuf:"bytes,2,opt,name=pagination,proto3" json:"pagination,omitempty"`
}

func (x *GetCommandsResponse) Reset() {
	*x = GetCommandsResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetCommandsResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetCommandsResponse) ProtoMessage() {}

func (x *GetCommandsResponse) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetCommandsResponse.ProtoReflect.Descriptor instead.
func (*GetCommandsResponse) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{1}
}

func (x *GetCommandsResponse) GetCommands() []*commandv1.Command {
	if x != nil {
		return x.Commands
	}
	return nil
}

func (x *GetCommandsResponse) GetPagination() *Pagination {
	if x != nil {
		return x.Pagination
	}
	return nil
}

// Get the requested command.
type GetCommandRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The id of the command.
	CommandId string `protobuf:"bytes,1,opt,name=command_id,json=commandId,proto3" json:"command_id,omitempty"`
}

func (x *GetCommandRequest) Reset() {
	*x = GetCommandRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetCommandRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetCommandRequest) ProtoMessage() {}

func (x *GetCommandRequest) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetCommandRequest.ProtoReflect.Descriptor instead.
func (*GetCommandRequest) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{2}
}

func (x *GetCommandRequest) GetCommandId() string {
	if x != nil {
		return x.CommandId
	}
	return ""
}

// Response to GetCommandRequest.
type GetCommandResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The requested command.
	Command *commandv1.Command `protobuf:"bytes,1,opt,name=command,proto3" json:"command,omitempty"`
	// The command config.
	Config *_struct.Struct `protobuf:"bytes,2,opt,name=config,proto3" json:"config,omitempty"`
}

func (x *GetCommandResponse) Reset() {
	*x = GetCommandResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetCommandResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetCommandResponse) ProtoMessage() {}

func (x *GetCommandResponse) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetCommandResponse.ProtoReflect.Descriptor instead.
func (*GetCommandResponse) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{3}
}

func (x *GetCommandResponse) GetCommand() *commandv1.Command {
	if x != nil {
		return x.Command
	}
	return nil
}

func (x *GetCommandResponse) GetConfig() *_struct.Struct {
	if x != nil {
		return x.Config
	}
	return nil
}

// Kill the requested command.
type KillCommandRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The id of the command.
	CommandId string `protobuf:"bytes,1,opt,name=command_id,json=commandId,proto3" json:"command_id,omitempty"`
}

func (x *KillCommandRequest) Reset() {
	*x = KillCommandRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *KillCommandRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*KillCommandRequest) ProtoMessage() {}

func (x *KillCommandRequest) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use KillCommandRequest.ProtoReflect.Descriptor instead.
func (*KillCommandRequest) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{4}
}

func (x *KillCommandRequest) GetCommandId() string {
	if x != nil {
		return x.CommandId
	}
	return ""
}

// Response to KillCommandRequest.
type KillCommandResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The requested command.
	Command *commandv1.Command `protobuf:"bytes,1,opt,name=command,proto3" json:"command,omitempty"`
}

func (x *KillCommandResponse) Reset() {
	*x = KillCommandResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *KillCommandResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*KillCommandResponse) ProtoMessage() {}

func (x *KillCommandResponse) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use KillCommandResponse.ProtoReflect.Descriptor instead.
func (*KillCommandResponse) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{5}
}

func (x *KillCommandResponse) GetCommand() *commandv1.Command {
	if x != nil {
		return x.Command
	}
	return nil
}

// Set the priority of the requested command.
type SetCommandPriorityRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The id of the command.
	CommandId string `protobuf:"bytes,1,opt,name=command_id,json=commandId,proto3" json:"command_id,omitempty"`
	// The new priority.
	Priority int32 `protobuf:"varint,2,opt,name=priority,proto3" json:"priority,omitempty"`
}

func (x *SetCommandPriorityRequest) Reset() {
	*x = SetCommandPriorityRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SetCommandPriorityRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SetCommandPriorityRequest) ProtoMessage() {}

func (x *SetCommandPriorityRequest) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SetCommandPriorityRequest.ProtoReflect.Descriptor instead.
func (*SetCommandPriorityRequest) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{6}
}

func (x *SetCommandPriorityRequest) GetCommandId() string {
	if x != nil {
		return x.CommandId
	}
	return ""
}

func (x *SetCommandPriorityRequest) GetPriority() int32 {
	if x != nil {
		return x.Priority
	}
	return 0
}

// Response to SetCommandPriorityRequest.
type SetCommandPriorityResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The requested command.
	Command *commandv1.Command `protobuf:"bytes,1,opt,name=command,proto3" json:"command,omitempty"`
}

func (x *SetCommandPriorityResponse) Reset() {
	*x = SetCommandPriorityResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SetCommandPriorityResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SetCommandPriorityResponse) ProtoMessage() {}

func (x *SetCommandPriorityResponse) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SetCommandPriorityResponse.ProtoReflect.Descriptor instead.
func (*SetCommandPriorityResponse) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{7}
}

func (x *SetCommandPriorityResponse) GetCommand() *commandv1.Command {
	if x != nil {
		return x.Command
	}
	return nil
}

// Request to launch a command.
type LaunchCommandRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Command config (JSON).
	Config *_struct.Struct `protobuf:"bytes,1,opt,name=config,proto3" json:"config,omitempty"`
	// Template name.
	TemplateName string `protobuf:"bytes,2,opt,name=template_name,json=templateName,proto3" json:"template_name,omitempty"`
	// The files to run with the command.
	Files []*utilv1.File `protobuf:"bytes,3,rep,name=files,proto3" json:"files,omitempty"`
	// Additional data.
	Data []byte `protobuf:"bytes,4,opt,name=data,proto3" json:"data,omitempty"`
}

func (x *LaunchCommandRequest) Reset() {
	*x = LaunchCommandRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *LaunchCommandRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LaunchCommandRequest) ProtoMessage() {}

func (x *LaunchCommandRequest) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LaunchCommandRequest.ProtoReflect.Descriptor instead.
func (*LaunchCommandRequest) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{8}
}

func (x *LaunchCommandRequest) GetConfig() *_struct.Struct {
	if x != nil {
		return x.Config
	}
	return nil
}

func (x *LaunchCommandRequest) GetTemplateName() string {
	if x != nil {
		return x.TemplateName
	}
	return ""
}

func (x *LaunchCommandRequest) GetFiles() []*utilv1.File {
	if x != nil {
		return x.Files
	}
	return nil
}

func (x *LaunchCommandRequest) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

// Response to LaunchCommandRequest.
type LaunchCommandResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The requested command.
	Command *commandv1.Command `protobuf:"bytes,1,opt,name=command,proto3" json:"command,omitempty"`
	// The config;
	Config *_struct.Struct `protobuf:"bytes,2,opt,name=config,proto3" json:"config,omitempty"`
	// If the requested slots exceeded the current max available.
	Warnings []LaunchWarning `protobuf:"varint,3,rep,packed,name=warnings,proto3,enum=determined.api.v1.LaunchWarning" json:"warnings,omitempty"`
}

func (x *LaunchCommandResponse) Reset() {
	*x = LaunchCommandResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_determined_api_v1_command_proto_msgTypes[9]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *LaunchCommandResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LaunchCommandResponse) ProtoMessage() {}

func (x *LaunchCommandResponse) ProtoReflect() protoreflect.Message {
	mi := &file_determined_api_v1_command_proto_msgTypes[9]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LaunchCommandResponse.ProtoReflect.Descriptor instead.
func (*LaunchCommandResponse) Descriptor() ([]byte, []int) {
	return file_determined_api_v1_command_proto_rawDescGZIP(), []int{9}
}

func (x *LaunchCommandResponse) GetCommand() *commandv1.Command {
	if x != nil {
		return x.Command
	}
	return nil
}

func (x *LaunchCommandResponse) GetConfig() *_struct.Struct {
	if x != nil {
		return x.Config
	}
	return nil
}

func (x *LaunchCommandResponse) GetWarnings() []LaunchWarning {
	if x != nil {
		return x.Warnings
	}
	return nil
}

var File_determined_api_v1_command_proto protoreflect.FileDescriptor

var file_determined_api_v1_command_proto_rawDesc = []byte{
	0x0a, 0x1f, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x61, 0x70, 0x69,
	0x2f, 0x76, 0x31, 0x2f, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x12, 0x11, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70,
	0x69, 0x2e, 0x76, 0x31, 0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74, 0x2e, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x1a, 0x22, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x61,
	0x70, 0x69, 0x2f, 0x76, 0x31, 0x2f, 0x70, 0x61, 0x67, 0x69, 0x6e, 0x61, 0x74, 0x69, 0x6f, 0x6e,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x23, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e,
	0x65, 0x64, 0x2f, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2f, 0x76, 0x31, 0x2f, 0x63, 0x6f,
	0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x1d, 0x64, 0x65, 0x74,
	0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x75, 0x74, 0x69, 0x6c, 0x2f, 0x76, 0x31, 0x2f,
	0x75, 0x74, 0x69, 0x6c, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x2c, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x63, 0x2d, 0x67, 0x65, 0x6e, 0x2d, 0x73, 0x77, 0x61, 0x67, 0x67, 0x65, 0x72, 0x2f, 0x6f,
	0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2f, 0x61, 0x6e, 0x6e, 0x6f, 0x74, 0x61, 0x74, 0x69, 0x6f,
	0x6e, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0xd5, 0x02, 0x0a, 0x12, 0x47, 0x65, 0x74,
	0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x73, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12,
	0x45, 0x0a, 0x07, 0x73, 0x6f, 0x72, 0x74, 0x5f, 0x62, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0e,
	0x32, 0x2c, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70,
	0x69, 0x2e, 0x76, 0x31, 0x2e, 0x47, 0x65, 0x74, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x73,
	0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x2e, 0x53, 0x6f, 0x72, 0x74, 0x42, 0x79, 0x52, 0x06,
	0x73, 0x6f, 0x72, 0x74, 0x42, 0x79, 0x12, 0x35, 0x0a, 0x08, 0x6f, 0x72, 0x64, 0x65, 0x72, 0x5f,
	0x62, 0x79, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x1a, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72,
	0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x31, 0x2e, 0x4f, 0x72, 0x64,
	0x65, 0x72, 0x42, 0x79, 0x52, 0x07, 0x6f, 0x72, 0x64, 0x65, 0x72, 0x42, 0x79, 0x12, 0x16, 0x0a,
	0x06, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x18, 0x03, 0x20, 0x01, 0x28, 0x05, 0x52, 0x06, 0x6f,
	0x66, 0x66, 0x73, 0x65, 0x74, 0x12, 0x14, 0x0a, 0x05, 0x6c, 0x69, 0x6d, 0x69, 0x74, 0x18, 0x04,
	0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6c, 0x69, 0x6d, 0x69, 0x74, 0x12, 0x14, 0x0a, 0x05, 0x75,
	0x73, 0x65, 0x72, 0x73, 0x18, 0x05, 0x20, 0x03, 0x28, 0x09, 0x52, 0x05, 0x75, 0x73, 0x65, 0x72,
	0x73, 0x12, 0x19, 0x0a, 0x08, 0x75, 0x73, 0x65, 0x72, 0x5f, 0x69, 0x64, 0x73, 0x18, 0x06, 0x20,
	0x03, 0x28, 0x05, 0x52, 0x07, 0x75, 0x73, 0x65, 0x72, 0x49, 0x64, 0x73, 0x22, 0x62, 0x0a, 0x06,
	0x53, 0x6f, 0x72, 0x74, 0x42, 0x79, 0x12, 0x17, 0x0a, 0x13, 0x53, 0x4f, 0x52, 0x54, 0x5f, 0x42,
	0x59, 0x5f, 0x55, 0x4e, 0x53, 0x50, 0x45, 0x43, 0x49, 0x46, 0x49, 0x45, 0x44, 0x10, 0x00, 0x12,
	0x0e, 0x0a, 0x0a, 0x53, 0x4f, 0x52, 0x54, 0x5f, 0x42, 0x59, 0x5f, 0x49, 0x44, 0x10, 0x01, 0x12,
	0x17, 0x0a, 0x13, 0x53, 0x4f, 0x52, 0x54, 0x5f, 0x42, 0x59, 0x5f, 0x44, 0x45, 0x53, 0x43, 0x52,
	0x49, 0x50, 0x54, 0x49, 0x4f, 0x4e, 0x10, 0x02, 0x12, 0x16, 0x0a, 0x12, 0x53, 0x4f, 0x52, 0x54,
	0x5f, 0x42, 0x59, 0x5f, 0x53, 0x54, 0x41, 0x52, 0x54, 0x5f, 0x54, 0x49, 0x4d, 0x45, 0x10, 0x04,
	0x22, 0xa2, 0x01, 0x0a, 0x13, 0x47, 0x65, 0x74, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x73,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x3a, 0x0a, 0x08, 0x63, 0x6f, 0x6d, 0x6d,
	0x61, 0x6e, 0x64, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x1e, 0x2e, 0x64, 0x65, 0x74,
	0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2e,
	0x76, 0x31, 0x2e, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x52, 0x08, 0x63, 0x6f, 0x6d, 0x6d,
	0x61, 0x6e, 0x64, 0x73, 0x12, 0x3d, 0x0a, 0x0a, 0x70, 0x61, 0x67, 0x69, 0x6e, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1d, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72,
	0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x31, 0x2e, 0x50, 0x61, 0x67,
	0x69, 0x6e, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x52, 0x0a, 0x70, 0x61, 0x67, 0x69, 0x6e, 0x61, 0x74,
	0x69, 0x6f, 0x6e, 0x3a, 0x10, 0x92, 0x41, 0x0d, 0x0a, 0x0b, 0xd2, 0x01, 0x08, 0x63, 0x6f, 0x6d,
	0x6d, 0x61, 0x6e, 0x64, 0x73, 0x22, 0x32, 0x0a, 0x11, 0x47, 0x65, 0x74, 0x43, 0x6f, 0x6d, 0x6d,
	0x61, 0x6e, 0x64, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1d, 0x0a, 0x0a, 0x63, 0x6f,
	0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09,
	0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x49, 0x64, 0x22, 0x99, 0x01, 0x0a, 0x12, 0x47, 0x65,
	0x74, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x12, 0x38, 0x0a, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x0b, 0x32, 0x1e, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x63,
	0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2e, 0x76, 0x31, 0x2e, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e,
	0x64, 0x52, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x12, 0x2f, 0x0a, 0x06, 0x63, 0x6f,
	0x6e, 0x66, 0x69, 0x67, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x67, 0x6f, 0x6f,
	0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x53, 0x74, 0x72,
	0x75, 0x63, 0x74, 0x52, 0x06, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x3a, 0x18, 0x92, 0x41, 0x15,
	0x0a, 0x13, 0xd2, 0x01, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0xd2, 0x01, 0x06, 0x63,
	0x6f, 0x6e, 0x66, 0x69, 0x67, 0x22, 0x33, 0x0a, 0x12, 0x4b, 0x69, 0x6c, 0x6c, 0x43, 0x6f, 0x6d,
	0x6d, 0x61, 0x6e, 0x64, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1d, 0x0a, 0x0a, 0x63,
	0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52,
	0x09, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x49, 0x64, 0x22, 0x4f, 0x0a, 0x13, 0x4b, 0x69,
	0x6c, 0x6c, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x12, 0x38, 0x0a, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x1e, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e,
	0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2e, 0x76, 0x31, 0x2e, 0x43, 0x6f, 0x6d, 0x6d, 0x61,
	0x6e, 0x64, 0x52, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x22, 0x56, 0x0a, 0x19, 0x53,
	0x65, 0x74, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x50, 0x72, 0x69, 0x6f, 0x72, 0x69, 0x74,
	0x79, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1d, 0x0a, 0x0a, 0x63, 0x6f, 0x6d, 0x6d,
	0x61, 0x6e, 0x64, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09, 0x63, 0x6f,
	0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x49, 0x64, 0x12, 0x1a, 0x0a, 0x08, 0x70, 0x72, 0x69, 0x6f, 0x72,
	0x69, 0x74, 0x79, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x08, 0x70, 0x72, 0x69, 0x6f, 0x72,
	0x69, 0x74, 0x79, 0x22, 0x56, 0x0a, 0x1a, 0x53, 0x65, 0x74, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e,
	0x64, 0x50, 0x72, 0x69, 0x6f, 0x72, 0x69, 0x74, 0x79, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x12, 0x38, 0x0a, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x1e, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e,
	0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2e, 0x76, 0x31, 0x2e, 0x43, 0x6f, 0x6d, 0x6d, 0x61,
	0x6e, 0x64, 0x52, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x22, 0xb0, 0x01, 0x0a, 0x14,
	0x4c, 0x61, 0x75, 0x6e, 0x63, 0x68, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x52, 0x65, 0x71,
	0x75, 0x65, 0x73, 0x74, 0x12, 0x2f, 0x0a, 0x06, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x53, 0x74, 0x72, 0x75, 0x63, 0x74, 0x52, 0x06, 0x63,
	0x6f, 0x6e, 0x66, 0x69, 0x67, 0x12, 0x23, 0x0a, 0x0d, 0x74, 0x65, 0x6d, 0x70, 0x6c, 0x61, 0x74,
	0x65, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x74, 0x65,
	0x6d, 0x70, 0x6c, 0x61, 0x74, 0x65, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x2e, 0x0a, 0x05, 0x66, 0x69,
	0x6c, 0x65, 0x73, 0x18, 0x03, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x18, 0x2e, 0x64, 0x65, 0x74, 0x65,
	0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x75, 0x74, 0x69, 0x6c, 0x2e, 0x76, 0x31, 0x2e, 0x46,
	0x69, 0x6c, 0x65, 0x52, 0x05, 0x66, 0x69, 0x6c, 0x65, 0x73, 0x12, 0x12, 0x0a, 0x04, 0x64, 0x61,
	0x74, 0x61, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0c, 0x52, 0x04, 0x64, 0x61, 0x74, 0x61, 0x22, 0xda,
	0x01, 0x0a, 0x15, 0x4c, 0x61, 0x75, 0x6e, 0x63, 0x68, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x38, 0x0a, 0x07, 0x63, 0x6f, 0x6d, 0x6d,
	0x61, 0x6e, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1e, 0x2e, 0x64, 0x65, 0x74, 0x65,
	0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2e, 0x63, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x2e, 0x76,
	0x31, 0x2e, 0x43, 0x6f, 0x6d, 0x6d, 0x61, 0x6e, 0x64, 0x52, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61,
	0x6e, 0x64, 0x12, 0x2f, 0x0a, 0x06, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x17, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x62, 0x75, 0x66, 0x2e, 0x53, 0x74, 0x72, 0x75, 0x63, 0x74, 0x52, 0x06, 0x63, 0x6f, 0x6e,
	0x66, 0x69, 0x67, 0x12, 0x3c, 0x0a, 0x08, 0x77, 0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x73, 0x18,
	0x03, 0x20, 0x03, 0x28, 0x0e, 0x32, 0x20, 0x2e, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e,
	0x65, 0x64, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x76, 0x31, 0x2e, 0x4c, 0x61, 0x75, 0x6e, 0x63, 0x68,
	0x57, 0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x52, 0x08, 0x77, 0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67,
	0x73, 0x3a, 0x18, 0x92, 0x41, 0x15, 0x0a, 0x13, 0xd2, 0x01, 0x07, 0x63, 0x6f, 0x6d, 0x6d, 0x61,
	0x6e, 0x64, 0xd2, 0x01, 0x06, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x2a, 0x5a, 0x0a, 0x0d, 0x4c,
	0x61, 0x75, 0x6e, 0x63, 0x68, 0x57, 0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x12, 0x1e, 0x0a, 0x1a,
	0x4c, 0x41, 0x55, 0x4e, 0x43, 0x48, 0x5f, 0x57, 0x41, 0x52, 0x4e, 0x49, 0x4e, 0x47, 0x5f, 0x55,
	0x4e, 0x53, 0x50, 0x45, 0x43, 0x49, 0x46, 0x49, 0x45, 0x44, 0x10, 0x00, 0x12, 0x29, 0x0a, 0x25,
	0x4c, 0x41, 0x55, 0x4e, 0x43, 0x48, 0x5f, 0x57, 0x41, 0x52, 0x4e, 0x49, 0x4e, 0x47, 0x5f, 0x43,
	0x55, 0x52, 0x52, 0x45, 0x4e, 0x54, 0x5f, 0x53, 0x4c, 0x4f, 0x54, 0x53, 0x5f, 0x45, 0x58, 0x43,
	0x45, 0x45, 0x44, 0x45, 0x44, 0x10, 0x01, 0x42, 0x35, 0x5a, 0x33, 0x67, 0x69, 0x74, 0x68, 0x75,
	0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64,
	0x2d, 0x61, 0x69, 0x2f, 0x64, 0x65, 0x74, 0x65, 0x72, 0x6d, 0x69, 0x6e, 0x65, 0x64, 0x2f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x70, 0x6b, 0x67, 0x2f, 0x61, 0x70, 0x69, 0x76, 0x31, 0x62, 0x06,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_determined_api_v1_command_proto_rawDescOnce sync.Once
	file_determined_api_v1_command_proto_rawDescData = file_determined_api_v1_command_proto_rawDesc
)

func file_determined_api_v1_command_proto_rawDescGZIP() []byte {
	file_determined_api_v1_command_proto_rawDescOnce.Do(func() {
		file_determined_api_v1_command_proto_rawDescData = protoimpl.X.CompressGZIP(file_determined_api_v1_command_proto_rawDescData)
	})
	return file_determined_api_v1_command_proto_rawDescData
}

var file_determined_api_v1_command_proto_enumTypes = make([]protoimpl.EnumInfo, 2)
var file_determined_api_v1_command_proto_msgTypes = make([]protoimpl.MessageInfo, 10)
var file_determined_api_v1_command_proto_goTypes = []interface{}{
	(LaunchWarning)(0),                 // 0: determined.api.v1.LaunchWarning
	(GetCommandsRequest_SortBy)(0),     // 1: determined.api.v1.GetCommandsRequest.SortBy
	(*GetCommandsRequest)(nil),         // 2: determined.api.v1.GetCommandsRequest
	(*GetCommandsResponse)(nil),        // 3: determined.api.v1.GetCommandsResponse
	(*GetCommandRequest)(nil),          // 4: determined.api.v1.GetCommandRequest
	(*GetCommandResponse)(nil),         // 5: determined.api.v1.GetCommandResponse
	(*KillCommandRequest)(nil),         // 6: determined.api.v1.KillCommandRequest
	(*KillCommandResponse)(nil),        // 7: determined.api.v1.KillCommandResponse
	(*SetCommandPriorityRequest)(nil),  // 8: determined.api.v1.SetCommandPriorityRequest
	(*SetCommandPriorityResponse)(nil), // 9: determined.api.v1.SetCommandPriorityResponse
	(*LaunchCommandRequest)(nil),       // 10: determined.api.v1.LaunchCommandRequest
	(*LaunchCommandResponse)(nil),      // 11: determined.api.v1.LaunchCommandResponse
	(OrderBy)(0),                       // 12: determined.api.v1.OrderBy
	(*commandv1.Command)(nil),          // 13: determined.command.v1.Command
	(*Pagination)(nil),                 // 14: determined.api.v1.Pagination
	(*_struct.Struct)(nil),             // 15: google.protobuf.Struct
	(*utilv1.File)(nil),                // 16: determined.util.v1.File
}
var file_determined_api_v1_command_proto_depIdxs = []int32{
	1,  // 0: determined.api.v1.GetCommandsRequest.sort_by:type_name -> determined.api.v1.GetCommandsRequest.SortBy
	12, // 1: determined.api.v1.GetCommandsRequest.order_by:type_name -> determined.api.v1.OrderBy
	13, // 2: determined.api.v1.GetCommandsResponse.commands:type_name -> determined.command.v1.Command
	14, // 3: determined.api.v1.GetCommandsResponse.pagination:type_name -> determined.api.v1.Pagination
	13, // 4: determined.api.v1.GetCommandResponse.command:type_name -> determined.command.v1.Command
	15, // 5: determined.api.v1.GetCommandResponse.config:type_name -> google.protobuf.Struct
	13, // 6: determined.api.v1.KillCommandResponse.command:type_name -> determined.command.v1.Command
	13, // 7: determined.api.v1.SetCommandPriorityResponse.command:type_name -> determined.command.v1.Command
	15, // 8: determined.api.v1.LaunchCommandRequest.config:type_name -> google.protobuf.Struct
	16, // 9: determined.api.v1.LaunchCommandRequest.files:type_name -> determined.util.v1.File
	13, // 10: determined.api.v1.LaunchCommandResponse.command:type_name -> determined.command.v1.Command
	15, // 11: determined.api.v1.LaunchCommandResponse.config:type_name -> google.protobuf.Struct
	0,  // 12: determined.api.v1.LaunchCommandResponse.warnings:type_name -> determined.api.v1.LaunchWarning
	13, // [13:13] is the sub-list for method output_type
	13, // [13:13] is the sub-list for method input_type
	13, // [13:13] is the sub-list for extension type_name
	13, // [13:13] is the sub-list for extension extendee
	0,  // [0:13] is the sub-list for field type_name
}

func init() { file_determined_api_v1_command_proto_init() }
func file_determined_api_v1_command_proto_init() {
	if File_determined_api_v1_command_proto != nil {
		return
	}
	file_determined_api_v1_pagination_proto_init()
	if !protoimpl.UnsafeEnabled {
		file_determined_api_v1_command_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*GetCommandsRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*GetCommandsResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*GetCommandRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*GetCommandResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*KillCommandRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*KillCommandResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SetCommandPriorityRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SetCommandPriorityResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*LaunchCommandRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_determined_api_v1_command_proto_msgTypes[9].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*LaunchCommandResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_determined_api_v1_command_proto_rawDesc,
			NumEnums:      2,
			NumMessages:   10,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_determined_api_v1_command_proto_goTypes,
		DependencyIndexes: file_determined_api_v1_command_proto_depIdxs,
		EnumInfos:         file_determined_api_v1_command_proto_enumTypes,
		MessageInfos:      file_determined_api_v1_command_proto_msgTypes,
	}.Build()
	File_determined_api_v1_command_proto = out.File
	file_determined_api_v1_command_proto_rawDesc = nil
	file_determined_api_v1_command_proto_goTypes = nil
	file_determined_api_v1_command_proto_depIdxs = nil
}

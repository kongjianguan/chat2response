import { v4 as uuidv4 } from 'uuid';
import type {
  ResponsesRequest,
  ChatCompletionRequest,
  ChatMessage,
  InputItem,
  ContentPart,
  Tool,
  ChatTool,
  OutputItem,
  ResponseObject,
  StreamEvent,
  ChatCompletionChunk,
  ToolCall,
} from './types';

const DEBUG = process.env.DEBUG === 'true';

function debug(...args: unknown[]) {
  if (DEBUG) {
    console.log('[Converter]', ...args);
  }
}

// ============================================
// Request Conversion: Responses API → Chat Completions
// ============================================

export function convertResponsesToChat(body: ResponsesRequest): ChatCompletionRequest {
  const { model, input, instructions, tools, tool_choice, stream, temperature, max_tokens, top_p, user, reasoning_effort } = body;
  
  const messages: ChatMessage[] = [];
  
  // Add system instructions if provided
  if (instructions) {
    messages.push({ role: 'system', content: instructions });
  }

  if (typeof input === 'string') {
    messages.push({ role: 'user', content: input });
  } else if (Array.isArray(input)) {
    let lastAssistantMsg: ChatMessage | null = null;

    for (const item of input) {
      if (item.type === 'message') {
        const role = item.role === 'developer' ? 'system' : item.role;
        const msg: ChatMessage = {
          role: role as any,
          content: extractTextContent(item.content),
        };
        
        if (role === 'assistant') {
          lastAssistantMsg = msg;
        } else {
          lastAssistantMsg = null;
        }
        messages.push(msg);
      } 
      else if (item.type === 'function_call') {
        // If we have a function call, it MUST be attached to an assistant message
        const toolCall: ToolCall = {
          id: item.call_id || `call_${uuidv4().replace(/-/g, '')}`,
          type: 'function',
          function: {
            name: item.name || '',
            arguments: item.arguments || '{}',
          },
        };

        if (lastAssistantMsg && lastAssistantMsg.role === 'assistant') {
          if (!lastAssistantMsg.tool_calls) lastAssistantMsg.tool_calls = [];
          lastAssistantMsg.tool_calls.push(toolCall);
        } else {
          // Create a new assistant message if none exists to hold the tool call
          const msg: ChatMessage = {
            role: 'assistant',
            content: '',
            tool_calls: [toolCall],
          };
          lastAssistantMsg = msg;
          messages.push(msg);
        }
      } 
      else if (item.type === 'function_call_output') {
        messages.push({
          role: 'tool',
          content: item.output || '',
          tool_call_id: item.call_id || '',
        });
        lastAssistantMsg = null;
      }
    }
  }

  // --- Sanitize tool_calls pairing ---
  // DeepSeek (and OpenAI spec) require:
  //   assistant (tool_calls) → tool (response) → [next non-tool]
  // Remove orphaned tool_calls and orphaned tool messages to avoid 400 errors.

  // Forward pass: remove tool_calls without matching tool responses
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== 'assistant' || !msg.tool_calls?.length) continue;

    const responseIds = new Set<string>();
    for (let j = i + 1; j < messages.length; j++) {
      if (messages[j].role !== 'tool') break;
      const tid = messages[j].tool_call_id;
      if (tid) responseIds.add(tid);
    }

    const valid = msg.tool_calls.filter(tc => responseIds.has(tc.id));
    if (valid.length === msg.tool_calls.length) continue;

    if (valid.length === 0 && typeof msg.content === 'string' && msg.content === '') {
      messages.splice(i, 1);
    } else if (valid.length > 0) {
      msg.tool_calls = valid;
    } else {
      delete msg.tool_calls;
    }
  }

  // Backward pass: remove tool messages without a preceding assistant.tool_calls
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== 'tool') continue;
    let matched = false;
    for (let j = i - 1; j >= 0; j--) {
      if (messages[j].role === 'tool') continue;
      if (
        messages[j].role === 'assistant' &&
        messages[j].tool_calls?.some(tc => tc.id === messages[i].tool_call_id)
      ) {
        matched = true;
      }
      break;
    }
    if (!matched) {
      messages.splice(i, 1);
    }
  }

  // Map xhigh → max for reasoning effort (Responses API uses xhigh, DeepSeek/opencode uses max)
  const mappedEffort = reasoning_effort === 'xhigh' ? 'max' : reasoning_effort;

  // Convert tools
  const chatTools: ChatTool[] | undefined = tools?.map(convertTool);
  
  return {
    model,
    messages,
    tools: chatTools,
    tool_choice: tool_choice as ChatCompletionRequest['tool_choice'],
    stream: stream ?? true,
    temperature,
    max_tokens,
    top_p,
    user,
    ...(mappedEffort ? { reasoning_effort: mappedEffort } : {}),
  };
}

function extractTextContent(content: string | ContentPart[] | undefined): string {
  if (!content) return '';
  if (typeof content === 'string') return content;
  
  return content
    .filter((part): part is ContentPart & { type: 'input_text' | 'output_text' } => 
      part.type === 'input_text' || part.type === 'output_text'
    )
    .map(part => part.text || '')
    .join('');
}

function convertTool(tool: Tool): ChatTool {
  // Handle built-in tools (web_search, code_interpreter, file_search)
  // Convert them to function calls for compatibility
  if (tool.type === 'web_search' || tool.type === 'code_interpreter' || tool.type === 'file_search') {
    return {
      type: 'function',
      function: {
        name: tool.name || tool.type,
        description: tool.description || `${tool.type} tool`,
        parameters: tool.parameters || {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'The search query',
            },
          },
          required: ['query'],
        },
      },
    };
  }
  
  // Standard function tool
  if (tool.function) {
    return {
      type: 'function',
      function: {
        name: tool.function.name,
        description: tool.function.description || '',
        parameters: tool.function.parameters || { type: 'object', properties: {} },
      },
    };
  }
  
  // Fallback for simple tool definitions
  return {
    type: 'function',
    function: {
      name: tool.name || 'unknown_tool',
      description: tool.description || '',
      parameters: tool.parameters || { type: 'object', properties: {} },
    },
  };
}

// ============================================
// Stream Conversion: Chat Completions → Responses API
// ============================================

interface StreamState {
  responseId: string;
  outputItemId: string;
  outputIndex: number;
  contentIndex: number;
  fullText: string;
  reasoningText: string;
  reasoningItemId: string;
  isFirstChunk: boolean;
  isOutputItemAdded: boolean;
  isContentPartAdded: boolean;
  isReasoningAdded: boolean;
  isReasoningDone: boolean;
  lastFinishReason: string | null;
  isCompleted: boolean;
  currentToolCall?: {
    id: string;
    name: string;
    arguments: string;
    outputIndex: number;
  };
  completedToolCalls: Array<{
    id: string;
    name: string;
    arguments: string;
    outputIndex: number;
  }>;
  toolCallOutputIndex: number;
}

export function createStreamState(model: string): StreamState {
  return {
    responseId: `resp_${uuidv4().replace(/-/g, '')}`,
    outputItemId: `msg_${uuidv4().replace(/-/g, '')}`,
    outputIndex: 0,
    contentIndex: 0,
    fullText: '',
    reasoningText: '',
    reasoningItemId: '',
    isFirstChunk: true,
    isOutputItemAdded: false,
    isContentPartAdded: false,
    isReasoningAdded: false,
    isReasoningDone: false,
    lastFinishReason: null,
    isCompleted: false,
    completedToolCalls: [],
    toolCallOutputIndex: 1,
  };
}

export function createInitialResponseObject(
  state: StreamState,
  model: string,
  input: ResponsesRequest['input']
): ResponseObject {
  return {
    id: state.responseId,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model,
    status: 'in_progress',
    input: typeof input === 'string' ? [{ type: 'message', role: 'user', content: input }] : input,
    output: [],
  };
}

// ============================================
// Event Creation Functions
// ============================================

export function createResponseCreatedEvent(response: ResponseObject): StreamEvent {
  return {
    type: 'response.created',
    response: { ...response, status: 'in_progress' },
  };
}

export function createResponseInProgressEvent(response: ResponseObject): StreamEvent {
  return {
    type: 'response.in_progress',
    response,
  };
}

export function createOutputItemAddedEvent(state: StreamState): StreamEvent {
  const item: OutputItem = {
    id: state.outputItemId,
    type: 'message',
    role: 'assistant',
    content: [],
  };
  
  return {
    type: 'response.output_item.added',
    output_index: state.outputIndex,
    item,
  };
}

export function createReasoningOutputItemAddedEvent(state: StreamState): StreamEvent {
  state.reasoningItemId = `reason_${uuidv4().replace(/-/g, '')}`;
  const item: OutputItem = {
    id: state.reasoningItemId,
    type: 'reasoning',
    summary: [],
  };

  const outputIndex = state.outputIndex++;

  return {
    type: 'response.output_item.added',
    output_index: outputIndex,
    item,
  };
}

export function createReasoningTextDeltaEvent(state: StreamState, delta: string): StreamEvent {
  return {
    type: 'response.reasoning_text.delta',
    item_id: state.reasoningItemId,
    output_index: state.outputIndex - 1,
    content_index: 0,
    delta,
  };
}

export function createContentPartAddedEvent(state: StreamState): StreamEvent {
  return {
    type: 'response.content_part.added',
    item_id: state.outputItemId,
    output_index: state.outputIndex,
    content_index: state.contentIndex,
    part: { type: 'output_text', text: '' },
  };
}

export function createOutputTextDeltaEvent(state: StreamState, delta: string): StreamEvent {
  return {
    type: 'response.output_text.delta',
    item_id: state.outputItemId,
    output_index: state.outputIndex,
    content_index: state.contentIndex,
    delta,
  };
}

export function createOutputTextDoneEvent(state: StreamState): StreamEvent {
  return {
    type: 'response.output_text.done',
    item_id: state.outputItemId,
    output_index: state.outputIndex,
    content_index: state.contentIndex,
    text: state.fullText,
  };
}

export function createContentPartDoneEvent(state: StreamState): StreamEvent {
  return {
    type: 'response.content_part.done',
    item_id: state.outputItemId,
    output_index: state.outputIndex,
    content_index: state.contentIndex,
    part: { type: 'output_text', text: state.fullText },
  };
}

export function createOutputItemDoneEvent(state: StreamState): StreamEvent {
  const item: OutputItem = {
    id: state.outputItemId,
    type: 'message',
    role: 'assistant',
    content: [{ type: 'output_text', text: state.fullText }],
  };
  
  return {
    type: 'response.output_item.done',
    output_index: state.outputIndex,
    item,
  };
}

export function createResponseCompletedEvent(
  state: StreamState,
  model: string,
  input: ResponsesRequest['input'],
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number }
): StreamEvent {
  const output: OutputItem[] = [];
  
  // Add reasoning output item if exists
  if (state.isReasoningAdded) {
    output.push({
      id: state.reasoningItemId || `reason_${uuidv4().replace(/-/g, '')}`,
      type: 'reasoning',
      content: [{ type: 'reasoning_text', text: state.reasoningText }],
      summary: [],
    });
  }

  // Add text output item
  if (state.fullText || !state.isReasoningAdded) {
    const outputItem: OutputItem = {
      id: state.outputItemId,
      type: 'message',
      role: 'assistant',
      content: [{ type: 'output_text', text: state.fullText }],
    };
    output.push(outputItem);
  }
  
  // Add tool call outputs
  for (const tc of state.completedToolCalls) {
    output.push({
      id: tc.id,
      type: 'function_call',
      name: tc.name,
      arguments: tc.arguments,
      call_id: tc.id,
    });
  }
  
  const response: ResponseObject = {
    id: state.responseId,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model,
    status: 'completed',
    input: typeof input === 'string' ? [{ type: 'message', role: 'user', content: input }] : input,
    output,
    usage,
  };
  
  return {
    type: 'response.completed',
    response,
  };
}

export function createResponseIncompleteEvent(
  state: StreamState,
  model: string,
  input: ResponsesRequest['input'],
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number }
): StreamEvent {
  const output: OutputItem[] = [];
  
  if (state.isReasoningAdded) {
    output.push({
      id: state.reasoningItemId || `reason_${uuidv4().replace(/-/g, '')}`,
      type: 'reasoning',
      content: [{ type: 'reasoning_text', text: state.reasoningText }],
      summary: [],
    });
  }
  if (state.fullText || !state.isReasoningAdded) {
    output.push({
      id: state.outputItemId,
      type: 'message',
      role: 'assistant',
      content: [{ type: 'output_text', text: state.fullText }],
    });
  }
  for (const tc of state.completedToolCalls) {
    output.push({
      id: tc.id,
      type: 'function_call',
      name: tc.name,
      arguments: tc.arguments,
      call_id: tc.id,
    });
  }

  const reason: 'max_tokens' | 'content_filter' =
    state.lastFinishReason === 'content_filter' ? 'content_filter' : 'max_tokens';

  const response: ResponseObject = {
    id: state.responseId,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model,
    status: 'incomplete',
    incomplete_details: { reason },
    input: typeof input === 'string' ? [{ type: 'message', role: 'user', content: input }] : input,
    output,
    usage,
  };
  
  return {
    type: 'response.incomplete',
    response,
  };
}

function createFunctionCallArgumentsDeltaEvent(
  outputIndex: number,
  itemId: string,
  delta: string
): StreamEvent {
  return {
    type: 'response.function_call_arguments.delta',
    output_index: outputIndex,
    item_id: itemId,
    delta,
  };
}

function createFunctionCallArgumentsDoneEvent(
  outputIndex: number,
  itemId: string,
  arguments_: string
): StreamEvent {
  return {
    type: 'response.function_call_arguments.done',
    output_index: outputIndex,
    item_id: itemId,
    arguments: arguments_,
  };
}

function createFunctionCallOutputItemAddedEvent(
  outputIndex: number,
  itemId: string,
  name: string
): StreamEvent {
  const item: OutputItem = {
    id: itemId,
    type: 'function_call',
    name,
    arguments: '',
    call_id: itemId,
  };
  return {
    type: 'response.output_item.added',
    output_index: outputIndex,
    item,
  };
}

function createFunctionCallOutputItemDoneEvent(
  outputIndex: number,
  itemId: string,
  name: string,
  arguments_: string
): StreamEvent {
  const item: OutputItem = {
    id: itemId,
    type: 'function_call',
    name,
    arguments: arguments_,
    call_id: itemId,
  };
  return {
    type: 'response.output_item.done',
    output_index: outputIndex,
    item,
  };
}

function formatSSE(event: StreamEvent): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

function* finalizeReasoning(state: StreamState): Generator<string> {
  if (!state.isReasoningAdded || state.isReasoningDone) return;
  state.isReasoningDone = true;

  const reasoningOutputIndex = state.outputIndex - 1;

  // reasoning_text.done
  yield formatSSE({
    type: 'response.reasoning_text.done',
    item_id: state.reasoningItemId,
    output_index: reasoningOutputIndex,
    content_index: 0,
    text: state.reasoningText,
  });

  // output_item.done for reasoning
  yield formatSSE({
    type: 'response.output_item.done',
    output_index: reasoningOutputIndex,
    item: {
      id: state.reasoningItemId,
      type: 'reasoning',
      content: [{ type: 'reasoning_text', text: state.reasoningText }],
      summary: [],
    } as unknown as OutputItem,
  });
}

// ============================================
// Main Stream Processing
// ============================================

export async function* streamChatToResponses(
  stream: ReadableStream<Uint8Array>,
  model: string,
  input: ResponsesRequest['input']
): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  const state = createStreamState(model);
  
  // Track buffer for incomplete lines
  let buffer = '';
  let responseObj: ResponseObject | null = null;
  let usage: { input_tokens: number; output_tokens: number; total_tokens: number } | undefined;
  
  // Helper function to send completion events
  const sendCompletionEvents = function* (): Generator<string> {
    if (state.isCompleted) return;

    // Finalize any pending tool call
    if (state.currentToolCall) {
      state.completedToolCalls.push({ ...state.currentToolCall });
      state.currentToolCall = undefined;
    }

    // 1. Finalize Reasoning if still open (stream ended without normal content)
    if (state.isReasoningAdded && !state.isReasoningDone) {
      yield* finalizeReasoning(state);
    }

    // 2. Finalize Message Output if added
    if (state.isOutputItemAdded) {
      if (state.fullText.length > 0) {
        yield formatSSE(createOutputTextDoneEvent(state));
      }
      if (state.isContentPartAdded) {
        yield formatSSE(createContentPartDoneEvent(state));
      }
      yield formatSSE(createOutputItemDoneEvent(state));
    }
    
    // 3. Finalize Tool Calls
    for (let i = 0; i < state.completedToolCalls.length; i++) {
      const tc = state.completedToolCalls[i];
      yield formatSSE(createFunctionCallArgumentsDoneEvent(tc.outputIndex, tc.id, tc.arguments));
      yield formatSSE(createFunctionCallOutputItemDoneEvent(tc.outputIndex, tc.id, tc.name, tc.arguments));
    }
    
    // 4. Send appropriate completion event based on finish_reason
    if (state.lastFinishReason === 'length' || state.lastFinishReason === 'content_filter') {
      yield formatSSE(createResponseIncompleteEvent(state, model, input, usage));
    } else {
      yield formatSSE(createResponseCompletedEvent(state, model, input, usage));
    }
    state.isCompleted = true;
  };
  
  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        // Send completion events before ending
        yield* sendCompletionEvents();
        break;
      }
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer
      
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;
        
        const data = trimmed.slice(5).trim();
        
        if (data === '[DONE]') {
          // Stream complete, send final events
          yield* sendCompletionEvents();
          continue;
        }
        
        try {
          const chunk: ChatCompletionChunk = JSON.parse(data);
          
          // Send initial events on first chunk
          if (state.isFirstChunk) {
            state.isFirstChunk = false;
            responseObj = createInitialResponseObject(state, model, input);
            
            // 1. response.created
            yield formatSSE(createResponseCreatedEvent(responseObj));
            
            // 2. response.in_progress
            yield formatSSE(createResponseInProgressEvent(responseObj));
          }
          
          // Process the chunk
          if (chunk.choices && chunk.choices.length > 0) {
            const choice = chunk.choices[0];
            const reasoningDelta = (choice.delta as any)?.reasoning_content;
            if (reasoningDelta) {
              // First reasoning chunk: open a reasoning output item
              if (!state.isReasoningAdded) {
                state.isReasoningAdded = true;
                yield formatSSE(createReasoningOutputItemAddedEvent(state));
              }
              state.reasoningText += reasoningDelta;
              yield formatSSE(createReasoningTextDeltaEvent(state, reasoningDelta));
              continue;
            }

            // Handle normal content delta
            let delta = choice.delta?.content;
            
            if (delta) {
              // First content after reasoning: close reasoning item
              if (state.isReasoningAdded && !state.isReasoningDone) {
                yield* finalizeReasoning(state);
              }

              // Ensure we have a message output item and content part if content starts
              if (!state.isOutputItemAdded) {
                yield formatSSE(createOutputItemAddedEvent(state));
                state.isOutputItemAdded = true;
                state.isContentPartAdded = true;
                yield formatSSE(createContentPartAddedEvent(state));
              }

              state.fullText += delta;
              yield formatSSE(createOutputTextDeltaEvent(state, delta));
            }
            
            // Handle tool calls
            if (choice.delta?.tool_calls) {
              for (const toolCall of choice.delta.tool_calls) {
                // Handle tool call initialization
                if (toolCall.id && toolCall.function?.name) {
                  // If there was a previous incomplete tool call, save it
                  if (state.currentToolCall) {
                    state.completedToolCalls.push({ ...state.currentToolCall });
                  }
                  
                  const callOutputIndex = state.toolCallOutputIndex;
                  state.toolCallOutputIndex++;
                  state.currentToolCall = {
                    id: toolCall.id,
                    name: toolCall.function.name,
                    arguments: toolCall.function.arguments || '',
                    outputIndex: callOutputIndex,
                  };
                  
                  // Emit output_item.added for this function call
                  yield formatSSE(createFunctionCallOutputItemAddedEvent(
                    callOutputIndex,
                    state.currentToolCall.id,
                    state.currentToolCall.name
                  ));
                } else if (toolCall.function?.arguments && state.currentToolCall) {
                  // Accumulate arguments
                  state.currentToolCall.arguments += toolCall.function.arguments;
                  
                  // Emit arguments delta
                  yield formatSSE(createFunctionCallArgumentsDeltaEvent(
                    state.toolCallOutputIndex - 1,
                    state.currentToolCall.id,
                    toolCall.function.arguments
                  ));
                }
              }
            }
            
            // Capture usage if provided
            if (chunk.usage) {
              usage = {
                input_tokens: chunk.usage.input_tokens || 0,
                output_tokens: chunk.usage.output_tokens || 0,
                total_tokens: chunk.usage.total_tokens || 0,
              };
            }
            
            // Handle finish reason
            if (choice.finish_reason) {
              state.lastFinishReason = choice.finish_reason;
              debug('Finish reason:', choice.finish_reason);
              
              // Finalize current tool call if present
              if (state.currentToolCall && choice.finish_reason === 'tool_calls') {
                state.completedToolCalls.push({ ...state.currentToolCall });
                state.currentToolCall = undefined;
              }
            }
          }
        } catch (e) {
          debug('Failed to parse chunk:', data, e);
        }
      }
    }
    
    // Send final events if not already sent
    yield* sendCompletionEvents();
    
    // Send [DONE]
    yield 'data: [DONE]\n\n';
    
  } finally {
    reader.releaseLock();
  }
}

// ============================================
// Non-streaming Response Conversion
// ============================================

export function convertChatToResponses(
  chatResponse: unknown,
  model: string,
  input: ResponsesRequest['input']
): ResponseObject {
  const chat = chatResponse as {
    id: string;
    choices: Array<{
      message: {
        content: string;
        tool_calls?: Array<{
          id: string;
          type: 'function';
          function: {
            name: string;
            arguments: string;
          };
        }>;
      };
      finish_reason: string;
    }>;
    usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  };
  
  const message = chat.choices[0]?.message;
  let content = message?.content || '';
  const toolCalls = message?.tool_calls;
  
  // Support reasoning_content for reasoning models (e.g., GLM-5)
  const reasoningContent = (message as unknown as Record<string, string> | undefined)?.reasoning_content;
  if (!content && reasoningContent) {
    content = reasoningContent;
  }
  
  const output: OutputItem[] = [];
  
  // Add text output
  if (content) {
    output.push({
      id: `msg_${uuidv4().replace(/-/g, '')}`,
      type: 'message',
      role: 'assistant',
      content: [{ type: 'output_text', text: content }],
    });
  }
  
  // Add function call outputs
  if (toolCalls) {
    for (const toolCall of toolCalls) {
      output.push({
        id: toolCall.id,
        type: 'function_call',
        name: toolCall.function.name,
        arguments: toolCall.function.arguments,
        call_id: toolCall.id,
      });
    }
  }
  
  return {
    id: `resp_${uuidv4().replace(/-/g, '')}`,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model,
    status: 'completed',
    input: typeof input === 'string' ? [{ type: 'message', role: 'user', content: input }] : input,
    output,
    usage: chat.usage ? {
      input_tokens: chat.usage.prompt_tokens,
      output_tokens: chat.usage.completion_tokens,
      total_tokens: chat.usage.total_tokens,
    } : undefined,
  };
}

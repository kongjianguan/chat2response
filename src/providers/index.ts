	import type { ProviderConfig, ProviderName, ChatCompletionRequest, ChatMessage } from '../types';

export const PROVIDERS: Record<ProviderName, ProviderConfig> = {
  glm: {
    name: 'GLM',
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    defaultModel: 'glm-5',
    models: ['glm-5'],
    supportsTools: false, // GLM does not support function calling well
    supportsStreaming: true,
    transformRequest: (req: ChatCompletionRequest): ChatCompletionRequest => {
      // GLM doesn't support tools, so we remove them
      if (req.tools?.length) {
        console.log('[GLM] Removing unsupported tools:', req.tools.length);
      }
      
      const transformed: ChatCompletionRequest = {
        ...req,
        // GLM uses a different model naming convention
        model: req.model?.startsWith('glm-') ? req.model : 'glm-5',
      };
      
      // Remove tool-related fields
      delete transformed.tools;
      delete transformed.tool_choice;
      
      // Flatten messages to simple text format for GLM
      if (transformed.messages) {
        transformed.messages = transformed.messages.map(msg => ({
          role: msg.role,
          content: typeof msg.content === 'string' 
            ? msg.content 
            : JSON.stringify(msg.content),
          // Preserve tool_call_id for multi-turn tool results
          ...(msg.tool_call_id ? { tool_call_id: msg.tool_call_id } : {}),
        }));
      }
      
      return transformed;
    },
  },
  
  kimi: {
    name: 'Kimi',
    baseUrl: 'https://api.moonshot.cn/v1',
    defaultModel: 'kimi-coding',
    models: ['kimi-coding', 'moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k'],
    supportsTools: true,
    supportsStreaming: true,
    transformRequest: (req: ChatCompletionRequest): ChatCompletionRequest => {
      const transformed: ChatCompletionRequest = { ...req };
      
      // Handle Kimi Coding Plan endpoint switch
      if (process.env.KIMI_CODING_PLAN === 'true') {
        (PROVIDERS.kimi as any).baseUrl = 'https://api.kimi.com/coding/v1';
      } else {
        (PROVIDERS.kimi as any).baseUrl = 'https://api.moonshot.cn/v1';
      }

      if (transformed.tools) {
        transformed.tools = transformed.tools.map(tool => ({
          ...tool,
          function: {
            ...tool.function,
            parameters: {
              type: 'object',
              properties: {},
              required: [],
              ...tool.function.parameters,
            },
          },
        }));
      }
      
      if (!transformed.model?.includes('kimi') && !transformed.model?.includes('moonshot')) {
        transformed.model = 'kimi-coding';
      }
      
      return transformed;
    },
  },
  
  deepseek: {
    name: 'DeepSeek',
    baseUrl: 'https://api.deepseek.com/v1',
    defaultModel: 'deepseek-chat',
    models: ['deepseek-chat', 'deepseek-reasoner'],
    supportsTools: true,
    supportsStreaming: true,
    // DeepSeek follows OpenAI format closely, minimal transformation needed
    transformRequest: (req: ChatCompletionRequest): ChatCompletionRequest => {
      // DeepSeek doesn't support some OpenAI-specific parameters
      const transformed: ChatCompletionRequest = { ...req };
      
      // Remove unsupported fields
      delete (transformed as Record<string, unknown>)['store'];
      
      // Use DeepSeek model if not specified
      if (!transformed.model?.startsWith('deepseek')) {
        transformed.model = 'deepseek-chat';
      }
      
      return transformed;
    },
  },
  
  minimax: {
    name: 'MiniMax',
    baseUrl: 'https://api.minimax.chat/v1', // 国内版 API
    defaultModel: 'minimax-2.7',
    models: ['minimax-2.7'],
    supportsTools: true,
    supportsStreaming: true,
    transformRequest: (req: ChatCompletionRequest): ChatCompletionRequest => {
      const transformed: ChatCompletionRequest = { ...req };
      
      // MiniMax uses different model naming
      if (!transformed.model?.includes('minimax')) {
        transformed.model = 'minimax-2.7';
      }
      
      // MiniMax requires specific message format
      if (transformed.messages) {
        transformed.messages = transformed.messages.map(msg => {
          // Ensure content is always a string
          const content = typeof msg.content === 'string' 
            ? msg.content 
            : JSON.stringify(msg.content);
          
          return {
            role: msg.role,
            content,
            // Preserve tool_call_id for multi-turn tool results
            ...(msg.tool_call_id ? { tool_call_id: msg.tool_call_id } : {}),
          };
        });
      }
      
      return transformed;
    },
  },

  opencode: {
    name: 'OpenCode',
    baseUrl: 'https://opencode.ai/zen/go/v1',
    defaultModel: '',
    models: [],
    supportsTools: true,
    supportsStreaming: true,
    // Generic OpenAI-compatible proxy — ensure reasoning_content is present on assistant messages
    // DeepSeek thinking mode requires it; but the Responses API bridge strips it
    transformRequest: (req: ChatCompletionRequest): ChatCompletionRequest => {
      if (!req.messages) return req;
      const PLACEHOLDER = '(reasoning omitted)';
      const transformed: ChatCompletionRequest = { ...req, messages: req.messages.map(msg => {
        if (msg.role !== 'assistant') return msg;
        const anyMsg = msg as unknown as Record<string, unknown>;
        let rc = anyMsg.reasoning_content;

        // Strip placeholder prefix when reasoning has real content beyond it
        if (typeof rc === 'string' && rc.startsWith(PLACEHOLDER) && rc.length > PLACEHOLDER.length) {
          rc = rc.slice(PLACEHOLDER.length).trimStart();
        }

        if (rc === undefined || rc === null || rc === '') {
          // DeepSeek requires reasoning_content on assistant msgs in thinking mode.
          // Our Responses API bridge loses it, so use a placeholder like deepseek-tui.
          const { reasoning_content: _, ...rest } = anyMsg;
          return { ...rest, reasoning_content: PLACEHOLDER } as unknown as ChatMessage;
        }
        // Return cleaned reasoning_content if it was modified
        if (rc !== anyMsg.reasoning_content) {
          const { reasoning_content: _, ...rest } = anyMsg;
          return { ...rest, reasoning_content: rc } as unknown as ChatMessage;
        }
        return msg;
      })};
      return transformed;
    },
  },
};

export function getProvider(name: ProviderName): ProviderConfig {
  const provider = PROVIDERS[name];
  if (!provider) {
    throw new Error(`Unknown provider: ${name}`);
  }
  return provider;
}

export function getCurrentProvider(): ProviderConfig {
  const defaultProvider = (process.env.DEFAULT_PROVIDER as ProviderName) || 'deepseek';
  return getProvider(defaultProvider);
}

export function getApiKey(providerName: ProviderName): string {
  const envVar = `${providerName.toUpperCase()}_API_KEY`;
  const apiKey = process.env[envVar];
  
  if (!apiKey) {
    throw new Error(`Missing API key for ${providerName}. Set ${envVar} environment variable.`);
  }
  
  return apiKey.trim();
}

export function transformRequest(
  providerName: ProviderName,
  request: ChatCompletionRequest
): ChatCompletionRequest {
  const provider = getProvider(providerName);
  
  if (provider.transformRequest) {
    return provider.transformRequest(request);
  }
  
  return request;
}

export function isProviderSupported(name: string): name is ProviderName {
  return name in PROVIDERS;
}

export function detectProviderFromModel(modelId: string): ProviderName | null {
  const modelLower = modelId.toLowerCase();
  if (modelLower.includes('glm')) return 'glm';
  if (modelLower.includes('kimi')) return 'kimi';
  if (modelLower.includes('deepseek')) return 'deepseek';
  if (modelLower.includes('minimax')) return 'minimax';
  // opencode is a generic proxy — not auto-detectable by model name
  return null;
}

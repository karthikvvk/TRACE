import type { IModelVendor } from '../IModelVendor';
import type { OpenAIAccessSchema } from '../../server/openai/openai.access';

import { ModelVendorOpenAI } from '../openai/openai.vendor';


export interface DLocalAIServiceSettings {
  localAIHost: string;  // use OpenAI-compatible non-default hosts (full origin path)
  localAIKey: string;   // use OpenAI-compatible API keys
  csf?: boolean;
}

export const ModelVendorLocalAI: IModelVendor<DLocalAIServiceSettings, OpenAIAccessSchema> = {
  id: 'localai',
  name: 'TRACE Agent',
  displayRank: 1,           // show first in the vendor list
  displayGroup: 'local',
  location: 'local',
  instanceLimit: 4,
  hasServerConfigKey: 'hasLlmLocalAIHost',
  hasServerConfigFn: (backendCapabilities) => {
    // this is to show the green mark on the vendor icon in the setup screen
    return backendCapabilities.hasLlmLocalAIHost || backendCapabilities.hasLlmLocalAIKey;
  },

  /// client-side-fetch ///
  csfAvailable: _csfLocalAIAvailable,

  // functions
  initializeSetup: () => ({
    // Pre-fill with the TRACE / Friday backend URL so Big-AGI works out of the box
    localAIHost: 'http://127.0.0.1:8000',
    localAIKey: '',
    csf: true,  // client-side-fetch: browser calls TRACE backend directly (no Next.js proxy)
  }),
  getTransportAccess: (partialSetup) => ({
    dialect: 'localai',
    // Default csf=true for TRACE; always enabled when the host is 127.0.0.1
    clientSideFetch: _csfLocalAIAvailable(partialSetup) && (partialSetup?.csf !== false),
    oaiKey: partialSetup?.localAIKey || '',
    oaiOrg: '',
    oaiHost: partialSetup?.localAIHost || 'http://127.0.0.1:8000',
    heliKey: '',
  }),

  // OpenAI transport ('localai' dialect in 'access')
  rpcUpdateModelsOrThrow: ModelVendorOpenAI.rpcUpdateModelsOrThrow,

};

function _csfLocalAIAvailable(_s?: Partial<DLocalAIServiceSettings>) {
  // CSF is always available for local vendors — the browser reaches TRACE directly.
  return true;
}

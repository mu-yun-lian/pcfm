// Data shapes mirroring the PCFM backend JSON (see src/pcfm/webapp.py + app.js).

export interface RecommendedQuestion {
  label?: string
  text: string
}

export interface CollectionState {
  status?: string
  message?: string
  mode?: string
  candidate_count?: number
  [key: string]: unknown
}

export interface Person {
  person_id: string
  name: string
  avatar?: string
  description?: string
  identity_note?: string
  focus_domain?: string
  aliases?: string[]
  language?: string
  is_demo?: boolean
  last_message?: string
  source_count?: number
  message_count?: number
  conversation_version?: number | null
  conversation_status?: string
  conversation_status_text?: string
  recommended_questions?: RecommendedQuestion[]
  collection?: CollectionState
  archived_at?: string
  version_count?: number
  [key: string]: unknown
}

export interface EvidenceItem {
  title: string
  event_id?: string
  speaker?: string
  date?: string
  locator?: string
  support_score?: number
  url?: string
}

export interface StructuredPrediction {
  speech_act?: { label?: string }
  stance?: { label?: string }
}

export interface ModelUsage {
  total_calls?: number
  status?: string
}

export interface RealityCandidate {
  comparison_candidate_id: string
  score: number
  question?: string
  answer?: string
  source_title?: string
  speaker?: string
  source_date?: string
  locator?: string
  [key: string]: unknown
}

export interface Comparison {
  status: string
  message_id: string
  predicted_answer?: string
  reality_candidates?: RealityCandidate[]
  selected_candidate_id?: string
  context_consistency?: string
  agreements?: string[]
  differences?: string[]
  notice?: string
  [key: string]: unknown
}

export interface Message {
  message_id: string
  role: 'user' | 'assistant'
  text: string
  status?: string
  answer_status?: string
  person_prediction_status?: string
  style_status?: string
  response_accuracy_status?: string
  confidence?: number
  evidence?: EvidenceItem[]
  uncertainties?: string[]
  comparison?: Comparison
  model_usage?: ModelUsage
  structured_prediction?: StructuredPrediction
  knowledge_source?: string
  model_kind?: string
  dialogue_model_provider?: string
  dialogue_model_id?: string
  feedback?: string
  [key: string]: unknown
}

export interface SourceCounts {
  confirmed: number
  pending: number
  final_holdout: number
  [key: string]: number
}

export interface ResponseEventCandidate {
  candidate_id: string
  trigger?: string
  actual_response?: string
  source_locator?: string
  speaker?: string
  review_status?: string
  [key: string]: unknown
}

export interface ResponseEvent {
  label_status?: string
  [key: string]: unknown
}

export interface Source {
  source_id: string
  title: string
  text_preview?: string
  speaker?: string
  speaker_scope?: string
  format?: string
  dataset_role?: string
  review_status?: string
  response_events?: ResponseEvent[]
  llm_response_event_candidates?: ResponseEventCandidate[]
  [key: string]: unknown
}

export interface SurfaceExtraction {
  status?: string
  [key: string]: unknown
}

export interface OptimizationCandidate {
  candidate_id: string
  status?: string
  active_version_before?: number | null
  validation_reasons?: string[]
  surface_extraction?: SurfaceExtraction
  new_version?: number
  [key: string]: unknown
}

export interface ConversationVersion {
  version: number
  reason?: string
  created_at?: string
  content_update_status?: string
  style_update_status?: string
  response_accuracy_status?: string
  validation_status?: string
  [key: string]: unknown
}

export interface PublicResponseModel {
  event_frame_count?: number
  value_atom_count?: number
  value_orientation_count?: number
  preference_structure_count?: number
  knowledge_claim_count?: number
  [key: string]: unknown
}

export interface ConversationProfile {
  language?: string
  aliases?: string[]
  collection?: CollectionState
  [key: string]: unknown
}

export interface Conversation {
  active_version?: number | null
  source_counts: SourceCounts
  messages: Message[]
  status?: string
  status_text?: string
  profile?: ConversationProfile
  public_response_model?: PublicResponseModel
  dialogue_model_ref?: string
  session_title?: string
  sources?: Source[]
  optimization_candidates?: OptimizationCandidate[]
  versions?: ConversationVersion[]
  metrics?: Record<string, unknown>
  [key: string]: unknown
}

export interface Session {
  session_id: string
  title?: string
  active?: boolean
  message_count?: number
  updated_at?: string
  [key: string]: unknown
}

export interface Job {
  job_id?: string
  status: string
  result?: any
  error_message?: string
  [key: string]: unknown
}

export interface ModelService {
  service_id: string
  display_name: string
  protocol?: string
  base_url?: string
  provider?: string
  api_key_configured?: boolean
  enabled_models?: string[]
  models?: string[]
  default_model?: string
  call_readiness?: string
  connection_status?: string
  last_probe_model?: string
  last_error?: string
  timeout_seconds?: number
  environment_key?: string
  capabilities?: { structured_output?: boolean }
  enabled?: boolean
  [key: string]: unknown
}

export interface ModelServicesState {
  services: ModelService[]
  roles: Record<string, string>
}

export interface Capabilities {
  public_search?: { available?: boolean }
  [key: string]: unknown
}

export interface ModelOption {
  ref: string
  label: string
  service: ModelService
  modelId: string
  ready: boolean
}

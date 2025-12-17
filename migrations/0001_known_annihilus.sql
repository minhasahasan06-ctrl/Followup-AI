CREATE TABLE "admin_totp_secrets" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"secret_name" varchar NOT NULL,
	"totp_secret" varchar NOT NULL,
	"enabled" boolean DEFAULT false,
	"setup_completed_at" timestamp,
	"setup_completed_by" varchar,
	"last_verified_at" timestamp,
	"last_verified_by" varchar,
	"verification_count" integer DEFAULT 0,
	"failed_attempts" integer DEFAULT 0,
	"locked_until" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "admin_totp_secrets_secret_name_unique" UNIQUE("secret_name")
);
--> statement-breakpoint
CREATE TABLE "agent_audit_logs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"actor_type" varchar NOT NULL,
	"actor_id" varchar NOT NULL,
	"actor_role" varchar,
	"action" varchar NOT NULL,
	"action_category" varchar NOT NULL,
	"object_type" varchar NOT NULL,
	"object_id" varchar NOT NULL,
	"patient_id" varchar,
	"conversation_id" varchar,
	"message_id" varchar,
	"phi_accessed" boolean DEFAULT false,
	"phi_categories" jsonb,
	"access_reason" varchar,
	"details" jsonb,
	"previous_state" jsonb,
	"new_state" jsonb,
	"ip_address" varchar,
	"user_agent" text,
	"session_id" varchar,
	"success" boolean DEFAULT true,
	"error_code" varchar,
	"error_message" text,
	"timestamp" timestamp DEFAULT now() NOT NULL,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "agent_conversations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"conversation_type" varchar DEFAULT 'patient_clona' NOT NULL,
	"participant1_type" varchar NOT NULL,
	"participant1_id" varchar NOT NULL,
	"participant2_type" varchar NOT NULL,
	"participant2_id" varchar NOT NULL,
	"additional_participants" jsonb,
	"patient_id" varchar,
	"doctor_id" varchar,
	"assignment_id" varchar,
	"title" varchar,
	"status" varchar DEFAULT 'active' NOT NULL,
	"message_count" integer DEFAULT 0,
	"unread_counts" jsonb DEFAULT '{}'::jsonb,
	"last_message_at" timestamp,
	"last_message_preview" text,
	"last_message_sender_id" varchar,
	"last_message_sender_role" varchar,
	"openai_thread_id" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "agent_memory" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"agent_id" varchar NOT NULL,
	"patient_id" varchar,
	"user_id" varchar,
	"conversation_id" varchar,
	"memory_type" varchar NOT NULL,
	"storage_type" varchar NOT NULL,
	"content" text NOT NULL,
	"summary" text,
	"embedding" jsonb,
	"embedding_model" varchar,
	"source_type" varchar,
	"source_id" varchar,
	"importance" numeric(3, 2) DEFAULT '0.5',
	"access_count" integer DEFAULT 0,
	"last_accessed_at" timestamp,
	"expires_at" timestamp,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "agent_messages" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"msg_id" varchar NOT NULL,
	"conversation_id" varchar NOT NULL,
	"from_type" varchar NOT NULL,
	"from_id" varchar NOT NULL,
	"sender_role" varchar DEFAULT 'patient' NOT NULL,
	"sender_name" varchar,
	"sender_avatar" varchar,
	"to_json" jsonb NOT NULL,
	"message_type" varchar NOT NULL,
	"content" text,
	"payload_json" jsonb,
	"tool_call_id" varchar,
	"tool_name" varchar,
	"tool_input" jsonb,
	"tool_output" jsonb,
	"tool_status" varchar,
	"requires_approval" boolean DEFAULT false,
	"approval_status" varchar,
	"approved_by" varchar,
	"approved_at" timestamp,
	"approval_notes" text,
	"delivered" boolean DEFAULT false,
	"delivered_at" timestamp,
	"read_at" timestamp,
	"contains_phi" boolean DEFAULT false,
	"phi_redacted" boolean DEFAULT false,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	CONSTRAINT "agent_messages_msg_id_unique" UNIQUE("msg_id")
);
--> statement-breakpoint
CREATE TABLE "agent_tasks" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"agent_id" varchar NOT NULL,
	"user_id" varchar,
	"conversation_id" varchar,
	"message_id" varchar,
	"task_type" varchar NOT NULL,
	"task_name" varchar,
	"scheduled_at" timestamp,
	"recurring_pattern" varchar,
	"timezone" varchar DEFAULT 'UTC',
	"status" varchar DEFAULT 'pending' NOT NULL,
	"priority" integer DEFAULT 5,
	"input_payload" jsonb,
	"output_result" jsonb,
	"error_message" text,
	"attempts" integer DEFAULT 0,
	"max_attempts" integer DEFAULT 3,
	"last_attempt_at" timestamp,
	"next_retry_at" timestamp,
	"worker_id" varchar,
	"started_at" timestamp,
	"completed_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "agent_tool_assignments" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"agent_id" varchar NOT NULL,
	"tool_id" varchar NOT NULL,
	"custom_config" jsonb,
	"is_enabled" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "agent_tools" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" varchar NOT NULL,
	"display_name" varchar NOT NULL,
	"description" text,
	"tool_type" varchar NOT NULL,
	"config_json" jsonb,
	"parameters_schema" jsonb,
	"required_permissions" jsonb,
	"allowed_roles" jsonb,
	"requires_approval" boolean DEFAULT false,
	"approval_role" varchar,
	"is_enabled" boolean DEFAULT true,
	"version" integer DEFAULT 1,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "agent_tools_name_unique" UNIQUE("name")
);
--> statement-breakpoint
CREATE TABLE "agents" (
	"id" varchar PRIMARY KEY NOT NULL,
	"name" varchar NOT NULL,
	"description" text,
	"agent_type" varchar NOT NULL,
	"target_role" varchar NOT NULL,
	"persona_json" jsonb,
	"memory_policy_json" jsonb,
	"openai_assistant_id" varchar,
	"openai_model" varchar DEFAULT 'gpt-4o',
	"is_active" boolean DEFAULT true,
	"version" integer DEFAULT 1,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ai_alert_rules" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"rule_name" varchar NOT NULL,
	"rule_category" varchar NOT NULL,
	"description" text,
	"thresholds" jsonb,
	"severity" varchar NOT NULL,
	"enabled" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "ai_alert_rules_rule_name_unique" UNIQUE("rule_name")
);
--> statement-breakpoint
CREATE TABLE "ai_engagement_metrics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"period_start" timestamp NOT NULL,
	"period_end" timestamp NOT NULL,
	"adherence_score" numeric(5, 2),
	"checkins_completed" integer DEFAULT 0,
	"checkins_expected" integer DEFAULT 0,
	"captures_completed" integer DEFAULT 0,
	"surveys_completed" integer DEFAULT 0,
	"engagement_score" numeric(5, 2),
	"engagement_trend" varchar,
	"engagement_drop_14d" numeric(5, 2),
	"avg_time_to_alert" integer,
	"alerts_generated" integer DEFAULT 0,
	"alerts_acknowledged" integer DEFAULT 0,
	"alerts_dismissed" integer DEFAULT 0,
	"current_streak" integer DEFAULT 0,
	"longest_streak" integer DEFAULT 0,
	"computed_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ai_health_alerts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"alert_type" varchar NOT NULL,
	"alert_category" varchar NOT NULL,
	"severity" varchar NOT NULL,
	"priority" integer NOT NULL,
	"escalation_probability" numeric(4, 3),
	"title" varchar NOT NULL,
	"message" text NOT NULL,
	"disclaimer" text DEFAULT 'This is an observational pattern alert. Not a diagnosis or medical opinion.' NOT NULL,
	"contributing_metrics" jsonb,
	"trigger_rule" varchar,
	"trigger_threshold" numeric(10, 4),
	"trigger_value" numeric(10, 4),
	"status" varchar DEFAULT 'new' NOT NULL,
	"acknowledged_by" varchar,
	"acknowledged_at" timestamp,
	"dismissed_by" varchar,
	"dismissed_at" timestamp,
	"dismiss_reason" text,
	"notified_patient" boolean DEFAULT false,
	"notified_clinician" boolean DEFAULT false,
	"sms_alert_sent" boolean DEFAULT false,
	"email_alert_sent" boolean DEFAULT false,
	"clinician_notes" text,
	"clinician_id" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ai_qol_metrics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"wellness_index" numeric(5, 2),
	"wellness_components" jsonb,
	"wellness_trend" varchar,
	"functional_status" numeric(5, 2),
	"functional_components" jsonb,
	"selfcare_score" numeric(5, 2),
	"selfcare_components" jsonb,
	"stability_score" numeric(5, 2),
	"behavior_patterns" jsonb,
	"recorded_at" timestamp NOT NULL,
	"computed_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ai_trend_metrics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"metric_name" varchar NOT NULL,
	"metric_category" varchar NOT NULL,
	"raw_value" numeric(10, 4) NOT NULL,
	"baseline_14d_mean" numeric(10, 4),
	"baseline_14d_std" numeric(10, 4),
	"z_score" numeric(6, 3),
	"z_score_severity" varchar,
	"slope_3d" numeric(8, 5),
	"slope_7d" numeric(8, 5),
	"slope_14d" numeric(8, 5),
	"slope_direction" varchar,
	"volatility_index" numeric(8, 4),
	"volatility_level" varchar,
	"composite_trend_score" numeric(5, 2),
	"recorded_at" timestamp NOT NULL,
	"computed_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "analysis_jobs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"job_type" varchar NOT NULL,
	"study_id" varchar,
	"cohort_id" varchar,
	"report_id" varchar,
	"config_json" jsonb,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"progress" integer DEFAULT 0,
	"current_phase" varchar,
	"started_at" timestamp,
	"completed_at" timestamp,
	"execution_time_ms" integer,
	"result_summary" jsonb,
	"error_message" text,
	"error_stack" text,
	"retry_count" integer DEFAULT 0,
	"trigger_source" varchar DEFAULT 'manual',
	"created_by" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "approval_queue" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"request_type" varchar NOT NULL,
	"requester_id" varchar NOT NULL,
	"requester_type" varchar NOT NULL,
	"approver_id" varchar,
	"approver_role" varchar DEFAULT 'doctor' NOT NULL,
	"patient_id" varchar,
	"conversation_id" varchar,
	"message_id" varchar,
	"tool_execution_id" varchar,
	"tool_name" varchar,
	"request_payload" jsonb NOT NULL,
	"request_summary" text,
	"urgency" varchar DEFAULT 'normal' NOT NULL,
	"risk_level" varchar,
	"risk_factors" jsonb,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"decision" varchar,
	"decision_by" varchar,
	"decision_at" timestamp,
	"decision_notes" text,
	"modified_payload" jsonb,
	"expires_at" timestamp,
	"reminder_sent_at" timestamp,
	"escalated_at" timestamp,
	"escalated_to" varchar,
	"execution_result" jsonb,
	"executed_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "behavior_alerts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"triggered_at" timestamp NOT NULL,
	"alert_type" varchar NOT NULL,
	"severity" varchar NOT NULL,
	"priority" integer NOT NULL,
	"title" varchar NOT NULL,
	"message" text NOT NULL,
	"source_risk_score_id" varchar,
	"source_trend_id" varchar,
	"email_sent" boolean DEFAULT false,
	"email_sent_at" timestamp,
	"sms_sent" boolean DEFAULT false,
	"sms_sent_at" timestamp,
	"dashboard_notified" boolean DEFAULT true,
	"acknowledged" boolean DEFAULT false,
	"acknowledged_at" timestamp,
	"acknowledged_by" varchar,
	"resolved" boolean DEFAULT false,
	"resolved_at" timestamp,
	"resolution_notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "behavior_checkins" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"scheduled_time" timestamp,
	"completed_at" timestamp,
	"response_latency_minutes" integer,
	"skipped" boolean DEFAULT false,
	"skip_reason" text,
	"symptom_severity" integer,
	"symptom_description" text,
	"pain_level" integer,
	"medication_taken" boolean DEFAULT false,
	"medication_skipped_reason" text,
	"session_duration_seconds" integer,
	"interaction_count" integer,
	"avoidance_language_detected" boolean DEFAULT false,
	"avoidance_phrases" jsonb,
	"sentiment_polarity" numeric(5, 3),
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "behavior_metrics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"date" timestamp NOT NULL,
	"checkin_time_consistency_score" numeric(5, 3),
	"checkin_completion_rate" numeric(5, 3),
	"avg_response_latency_minutes" numeric(8, 2),
	"skipped_checkins_count" integer DEFAULT 0,
	"routine_deviation_score" numeric(5, 3),
	"medication_adherence_rate" numeric(5, 3),
	"medication_skips_count" integer DEFAULT 0,
	"app_engagement_duration_minutes" numeric(8, 2),
	"app_open_count" integer DEFAULT 0,
	"avoidance_patterns_detected" boolean DEFAULT false,
	"avoidance_count" integer DEFAULT 0,
	"avoidance_phrases" jsonb,
	"avg_sentiment_polarity" numeric(5, 3),
	"sentiment_trend_slope" numeric(8, 5),
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "behavior_risk_scores" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"calculated_at" timestamp NOT NULL,
	"behavioral_risk" numeric(5, 2) NOT NULL,
	"digital_biomarker_risk" numeric(5, 2) NOT NULL,
	"cognitive_risk" numeric(5, 2) NOT NULL,
	"sentiment_risk" numeric(5, 2) NOT NULL,
	"composite_risk" numeric(5, 2) NOT NULL,
	"risk_level" varchar NOT NULL,
	"model_type" varchar DEFAULT 'transformer_xgboost_ensemble',
	"model_version" varchar,
	"feature_contributions" jsonb,
	"top_risk_factors" jsonb,
	"prediction_confidence" numeric(5, 3),
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "chat_symptoms" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"session_id" varchar NOT NULL,
	"message_id" varchar,
	"raw_text" text NOT NULL,
	"extracted_json" jsonb NOT NULL,
	"confidence" numeric(3, 2),
	"extraction_model" varchar DEFAULT 'gpt-4o',
	"symptom_checkin_id" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "clinician_notes" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_id" varchar NOT NULL,
	"clinician_id" varchar NOT NULL,
	"note" text NOT NULL,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "clinician_workload_metrics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"clinician_id" varchar NOT NULL,
	"period_start" timestamp NOT NULL,
	"period_end" timestamp NOT NULL,
	"alerts_received" integer DEFAULT 0,
	"alerts_acknowledged" integer DEFAULT 0,
	"alerts_dismissed" integer DEFAULT 0,
	"alerts_escalated" integer DEFAULT 0,
	"avg_response_time_seconds" integer,
	"manual_checks_avoided" integer DEFAULT 0,
	"baseline_manual_checks" integer DEFAULT 0,
	"workload_reduction_percent" numeric(5, 2),
	"computed_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "cognitive_tests" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"test_type" varchar NOT NULL,
	"started_at" timestamp NOT NULL,
	"completed_at" timestamp,
	"duration_seconds" integer,
	"reaction_time_ms" integer,
	"tapping_speed" numeric(6, 2),
	"error_rate" numeric(5, 3),
	"memory_score" numeric(5, 3),
	"pattern_recall_accuracy" numeric(5, 3),
	"instruction_accuracy" numeric(5, 3),
	"raw_results" jsonb,
	"baseline_deviation" numeric(6, 3),
	"anomaly_detected" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "condition_trigger_mappings" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"condition_code" varchar NOT NULL,
	"condition_name" varchar NOT NULL,
	"factor_type" varchar NOT NULL,
	"factor_name" varchar NOT NULL,
	"trigger_threshold" numeric(10, 4),
	"critical_threshold" numeric(10, 4),
	"base_weight" numeric(5, 4) DEFAULT '0.5',
	"impact_direction" varchar NOT NULL,
	"clinical_evidence" text,
	"recommendations" jsonb,
	"is_active" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "daily_followup_assignments" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"template_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"study_id" varchar,
	"frequency" varchar DEFAULT 'daily',
	"start_date" timestamp NOT NULL,
	"end_date" timestamp,
	"is_active" boolean DEFAULT true,
	"notify_at" varchar,
	"reminder_enabled" boolean DEFAULT true,
	"created_by" varchar NOT NULL,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "daily_followup_responses" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"assignment_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"response_date" timestamp NOT NULL,
	"responses_json" jsonb NOT NULL,
	"completed_at" timestamp,
	"is_complete" boolean DEFAULT false,
	"overall_score" numeric(5, 2),
	"symptom_score" numeric(5, 2),
	"mood_score" numeric(5, 2),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "daily_followup_templates" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" varchar NOT NULL,
	"description" text,
	"questions_json" jsonb NOT NULL,
	"estimated_duration" integer,
	"category" varchar,
	"is_active" boolean DEFAULT true,
	"version" integer DEFAULT 1,
	"created_by" varchar NOT NULL,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "data_snapshots" (
	"id" varchar PRIMARY KEY NOT NULL,
	"created_at" timestamp DEFAULT now(),
	"table_checksums" jsonb,
	"row_counts" jsonb,
	"date_range" jsonb,
	"created_by" varchar NOT NULL,
	"description" text,
	"content_hash" varchar
);
--> statement-breakpoint
CREATE TABLE "deterioration_trends" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"detected_at" timestamp NOT NULL,
	"trend_type" varchar NOT NULL,
	"severity" varchar NOT NULL,
	"trend_start_date" timestamp,
	"trend_duration_days" integer,
	"trend_slope" numeric(10, 5),
	"z_score" numeric(6, 3),
	"p_value" numeric(10, 8),
	"confidence_level" numeric(5, 3),
	"affected_metrics" jsonb,
	"metric_values" jsonb,
	"clinical_significance" text,
	"recommended_actions" jsonb,
	"alert_generated" boolean DEFAULT false,
	"alert_id" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "device_data_audit_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"actor_id" varchar NOT NULL,
	"actor_type" varchar NOT NULL,
	"actor_role" varchar,
	"action" varchar NOT NULL,
	"action_category" varchar NOT NULL,
	"resource_type" varchar NOT NULL,
	"resource_id" varchar,
	"patient_id" varchar,
	"event_details" jsonb,
	"ip_address" varchar,
	"user_agent" text,
	"session_id" varchar,
	"request_id" varchar,
	"success" boolean DEFAULT true,
	"error_message" text,
	"phi_accessed" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "device_health" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"device_connection_id" varchar NOT NULL,
	"status" varchar DEFAULT 'unknown' NOT NULL,
	"status_message" text,
	"last_seen_at" timestamp,
	"battery_level" integer,
	"battery_status" varchar,
	"estimated_battery_life" varchar,
	"signal_strength" integer,
	"connection_type" varchar,
	"last_connection_at" timestamp,
	"connection_errors" integer DEFAULT 0,
	"firmware_version" varchar,
	"firmware_update_available" boolean DEFAULT false,
	"latest_firmware_version" varchar,
	"sync_success_rate" numeric(5, 2),
	"avg_sync_latency" integer,
	"last_successful_sync" timestamp,
	"consecutive_failures" integer DEFAULT 0,
	"data_quality_score" integer,
	"missing_data_types" jsonb,
	"data_gaps" jsonb,
	"health_alerts" jsonb,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "device_models" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"vendor_id" varchar NOT NULL,
	"vendor_name" varchar NOT NULL,
	"model_name" varchar NOT NULL,
	"model_number" varchar,
	"device_type" varchar NOT NULL,
	"device_category" varchar NOT NULL,
	"capabilities" jsonb NOT NULL,
	"pairing_methods" jsonb NOT NULL,
	"api_config" jsonb,
	"baa_required" boolean DEFAULT false,
	"baa_status" varchar,
	"hipaa_compliant" boolean DEFAULT false,
	"fda_cleared" boolean DEFAULT false,
	"fda_clearance_number" varchar,
	"setup_instructions" text,
	"troubleshooting_guide" text,
	"docs_url" varchar,
	"support_url" varchar,
	"is_active" boolean DEFAULT true,
	"is_public_api" boolean DEFAULT false,
	"requires_partnership" boolean DEFAULT false,
	"image_url" varchar,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "device_pairing_sessions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"device_model_id" varchar,
	"device_type" varchar NOT NULL,
	"vendor_id" varchar NOT NULL,
	"pairing_method" varchar NOT NULL,
	"session_status" varchar DEFAULT 'initiated' NOT NULL,
	"oauth_state" varchar,
	"oauth_code_verifier" text,
	"oauth_redirect_uri" varchar,
	"ble_device_id" varchar,
	"ble_service_uuid" varchar,
	"ble_pairing_code" varchar,
	"qr_code_token" varchar,
	"qr_code_expires_at" timestamp,
	"result_device_connection_id" varchar,
	"result_vendor_account_id" varchar,
	"error_code" varchar,
	"error_message" text,
	"error_details" jsonb,
	"consent_captured" boolean DEFAULT false,
	"consent_timestamp" timestamp,
	"consent_version" varchar,
	"consented_data_types" jsonb,
	"started_at" timestamp DEFAULT now(),
	"expires_at" timestamp,
	"completed_at" timestamp,
	"ip_address" varchar,
	"user_agent" text,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "device_readings" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"device_type" varchar NOT NULL,
	"device_brand" varchar,
	"device_model" varchar,
	"source" varchar DEFAULT 'manual' NOT NULL,
	"wearable_integration_id" varchar,
	"recorded_at" timestamp DEFAULT now() NOT NULL,
	"bp_systolic" integer,
	"bp_diastolic" integer,
	"bp_pulse" integer,
	"bp_irregular_heartbeat" boolean,
	"bp_body_position" varchar,
	"bp_arm_used" varchar,
	"glucose_value" numeric(5, 1),
	"glucose_context" varchar,
	"glucose_unit" varchar DEFAULT 'mg/dL',
	"weight" numeric(5, 2),
	"weight_unit" varchar DEFAULT 'kg',
	"bmi" numeric(4, 1),
	"body_fat_percentage" numeric(4, 1),
	"muscle_mass" numeric(5, 2),
	"bone_mass" numeric(4, 2),
	"water_percentage" numeric(4, 1),
	"visceral_fat" integer,
	"metabolic_age" integer,
	"temperature" numeric(4, 1),
	"temperature_unit" varchar DEFAULT 'F',
	"temperature_location" varchar,
	"stethoscope_audio_url" varchar,
	"stethoscope_location" varchar,
	"heart_sounds_analysis" jsonb,
	"lung_sounds_analysis" jsonb,
	"heart_rate" integer,
	"resting_heart_rate" integer,
	"hrv" integer,
	"hrv_sdnn" integer,
	"ecg_data" jsonb,
	"afib_detected" boolean,
	"irregular_rhythm_alert" boolean,
	"heart_rate_zones" jsonb,
	"spo2" integer,
	"spo2_min" integer,
	"respiratory_rate" integer,
	"sleep_duration" integer,
	"sleep_deep_minutes" integer,
	"sleep_rem_minutes" integer,
	"sleep_light_minutes" integer,
	"sleep_awake_minutes" integer,
	"sleep_score" integer,
	"sleep_efficiency" numeric(4, 1),
	"sleep_consistency" integer,
	"sleep_debt" integer,
	"sleep_need" integer,
	"recovery_score" integer,
	"readiness_score" integer,
	"body_battery" integer,
	"strain_score" numeric(3, 1),
	"stress_score" integer,
	"skin_temperature" numeric(4, 2),
	"skin_temp_unit" varchar DEFAULT 'C',
	"steps" integer,
	"active_minutes" integer,
	"calories_burned" integer,
	"distance_meters" integer,
	"floors_climbed" integer,
	"standing_hours" integer,
	"vo2_max" numeric(4, 1),
	"training_load" integer,
	"training_status" varchar,
	"training_readiness" integer,
	"fitness_age" integer,
	"running_dynamics" jsonb,
	"lactate_threshold" integer,
	"performance_condition" integer,
	"cycle_day" integer,
	"cycle_phase" varchar,
	"predicted_ovulation" timestamp,
	"period_logged" boolean,
	"fall_detected" boolean,
	"fall_timestamp" timestamp,
	"emergency_sos_triggered" boolean,
	"route_to_hypertension" boolean DEFAULT false,
	"route_to_diabetes" boolean DEFAULT false,
	"route_to_cardiovascular" boolean DEFAULT false,
	"route_to_respiratory" boolean DEFAULT false,
	"route_to_sleep" boolean DEFAULT false,
	"route_to_mental_health" boolean DEFAULT false,
	"route_to_fitness" boolean DEFAULT false,
	"processed_for_alerts" boolean DEFAULT false,
	"alerts_generated" jsonb,
	"contributed_to_ml_training" boolean DEFAULT false,
	"notes" text,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "device_sync_jobs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"job_type" varchar NOT NULL,
	"vendor_account_id" varchar,
	"device_connection_id" varchar,
	"user_id" varchar,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"priority" integer DEFAULT 5,
	"scheduled_for" timestamp,
	"started_at" timestamp,
	"completed_at" timestamp,
	"attempts" integer DEFAULT 0,
	"max_attempts" integer DEFAULT 3,
	"next_retry_at" timestamp,
	"records_processed" integer DEFAULT 0,
	"records_failed" integer DEFAULT 0,
	"data_types" jsonb,
	"date_range" jsonb,
	"error_code" varchar,
	"error_message" text,
	"error_stack" text,
	"worker_id" varchar,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "digital_biomarkers" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"date" timestamp NOT NULL,
	"daily_step_count" integer,
	"step_trend_7day" numeric(8, 2),
	"activity_burst_count" integer,
	"sedentary_duration_minutes" integer,
	"movement_variability_score" numeric(5, 3),
	"circadian_rhythm_stability" numeric(5, 3),
	"sleep_wake_irregularity_minutes" integer,
	"daily_peak_activity_time" varchar,
	"phone_usage_irregularity" numeric(5, 3),
	"night_phone_interaction_count" integer,
	"screen_on_duration_minutes" integer,
	"mobility_drop_detected" boolean DEFAULT false,
	"mobility_change_percent" numeric(6, 2),
	"accelerometer_std_dev" numeric(10, 5),
	"accelerometer_mean_magnitude" numeric(10, 5),
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "doctor_emails" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"integration_id" varchar NOT NULL,
	"provider_message_id" varchar NOT NULL,
	"thread_id" varchar,
	"subject" varchar,
	"from_email" varchar NOT NULL,
	"from_name" varchar,
	"to_emails" jsonb,
	"cc_emails" jsonb,
	"snippet" text,
	"body_plain" text,
	"body_html" text,
	"is_read" boolean DEFAULT false,
	"is_starred" boolean DEFAULT false,
	"labels" jsonb,
	"linked_patient_id" varchar,
	"patient_link_confidence" numeric(3, 2),
	"ai_category" varchar,
	"ai_priority" varchar,
	"ai_summary" text,
	"ai_suggested_reply" text,
	"ai_extracted_info" jsonb,
	"has_been_replied" boolean DEFAULT false,
	"replied_at" timestamp,
	"received_at" timestamp NOT NULL,
	"synced_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "doctor_integrations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"integration_type" varchar NOT NULL,
	"status" varchar DEFAULT 'disconnected' NOT NULL,
	"last_sync_at" timestamp,
	"last_error_at" timestamp,
	"last_error_message" text,
	"access_token" text,
	"refresh_token" text,
	"token_expires_at" timestamp,
	"token_scope" text,
	"provider_account_id" varchar,
	"provider_account_email" varchar,
	"twilio_phone_number" varchar,
	"twilio_account_sid" varchar,
	"twilio_api_key" varchar,
	"twilio_api_secret" text,
	"whatsapp_business_id" varchar,
	"whatsapp_phone_number_id" varchar,
	"whatsapp_display_number" varchar,
	"sync_enabled" boolean DEFAULT true,
	"sync_frequency_minutes" integer DEFAULT 5,
	"auto_reply_enabled" boolean DEFAULT false,
	"email_labels_to_sync" jsonb,
	"email_auto_categorization_enabled" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "doctor_patient_assignments" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"status" varchar DEFAULT 'active' NOT NULL,
	"assignment_source" varchar NOT NULL,
	"source_reference_id" varchar,
	"patient_consented" boolean DEFAULT false,
	"consented_at" timestamp,
	"consent_method" varchar,
	"is_primary_care_provider" boolean DEFAULT false,
	"access_scope" varchar DEFAULT 'full',
	"access_notes" text,
	"revoked_at" timestamp,
	"revoked_by" varchar,
	"revocation_reason" text,
	"assigned_by" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "doctor_patient_consent_permissions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"assignment_id" varchar NOT NULL,
	"doctor_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"share_health_data" boolean DEFAULT false NOT NULL,
	"confidentiality_agreed" boolean DEFAULT false NOT NULL,
	"share_medical_files" boolean DEFAULT false NOT NULL,
	"share_medications" boolean DEFAULT true NOT NULL,
	"share_ai_messages" boolean DEFAULT false NOT NULL,
	"share_doctor_messages" boolean DEFAULT true NOT NULL,
	"share_daily_followups" boolean DEFAULT true NOT NULL,
	"share_health_alerts" boolean DEFAULT true NOT NULL,
	"share_behavioral_insights" boolean DEFAULT false NOT NULL,
	"share_pain_tracking" boolean DEFAULT true NOT NULL,
	"share_vital_signs" boolean DEFAULT true NOT NULL,
	"consent_epidemiological_research" boolean DEFAULT false NOT NULL,
	"terms_version" varchar DEFAULT '1.0' NOT NULL,
	"terms_agreed_at" timestamp,
	"digital_signature" text,
	"signature_method" varchar,
	"consent_ip_address" varchar,
	"consent_user_agent" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	"modified_by" varchar
);
--> statement-breakpoint
CREATE TABLE "doctor_whatsapp_messages" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"integration_id" varchar NOT NULL,
	"wa_message_id" varchar NOT NULL,
	"wa_conversation_id" varchar,
	"direction" varchar NOT NULL,
	"from_number" varchar NOT NULL,
	"to_number" varchar NOT NULL,
	"contact_name" varchar,
	"message_type" varchar NOT NULL,
	"text_content" text,
	"media_url" varchar,
	"media_mime_type" varchar,
	"linked_patient_id" varchar,
	"ai_category" varchar,
	"ai_priority" varchar,
	"ai_suggested_reply" text,
	"ai_extracted_info" jsonb,
	"status" varchar DEFAULT 'received' NOT NULL,
	"replied_at" timestamp,
	"received_at" timestamp NOT NULL,
	"synced_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "dosage_change_requests" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"medication_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"request_type" varchar NOT NULL,
	"current_dosage" varchar NOT NULL,
	"requested_dosage" varchar NOT NULL,
	"current_frequency" varchar NOT NULL,
	"requested_frequency" varchar NOT NULL,
	"scheduled_change_date" timestamp,
	"request_reason" text NOT NULL,
	"additional_notes" text,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"reviewed_by_doctor_id" varchar,
	"reviewed_at" timestamp,
	"doctor_notes" text,
	"doctor_notified_at" timestamp,
	"patient_notified_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "entity_embeddings" (
	"id" varchar PRIMARY KEY NOT NULL,
	"entity_type" varchar NOT NULL,
	"method" varchar NOT NULL,
	"embedding_dim" integer NOT NULL,
	"n_entities" integer NOT NULL,
	"entity_mapping" jsonb,
	"training_loss" numeric(10, 6),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "environmental_alerts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"alert_type" varchar NOT NULL,
	"triggered_by" varchar NOT NULL,
	"severity" varchar NOT NULL,
	"priority" integer NOT NULL,
	"title" varchar NOT NULL,
	"message" text NOT NULL,
	"recommendations" jsonb,
	"trigger_value" numeric(10, 4),
	"threshold_value" numeric(10, 4),
	"percent_over_threshold" numeric(6, 2),
	"risk_score_id" varchar,
	"forecast_id" varchar,
	"snapshot_id" varchar,
	"status" varchar DEFAULT 'active',
	"acknowledged_at" timestamp,
	"resolved_at" timestamp,
	"expires_at" timestamp,
	"push_notification_sent" boolean DEFAULT false,
	"sms_notification_sent" boolean DEFAULT false,
	"email_notification_sent" boolean DEFAULT false,
	"notification_sent_at" timestamp,
	"was_helpful" boolean,
	"user_feedback" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "environmental_data_snapshots" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"zip_code" varchar(10) NOT NULL,
	"measured_at" timestamp NOT NULL,
	"temperature" numeric(6, 2),
	"feels_like" numeric(6, 2),
	"humidity" numeric(5, 2),
	"pressure" numeric(7, 2),
	"wind_speed" numeric(6, 2),
	"wind_direction" integer,
	"precipitation" numeric(6, 2),
	"uv_index" numeric(4, 2),
	"cloud_cover" integer,
	"visibility" numeric(8, 2),
	"aqi" integer,
	"aqi_category" varchar,
	"pm25" numeric(7, 3),
	"pm10" numeric(7, 3),
	"ozone" numeric(7, 3),
	"no2" numeric(7, 3),
	"so2" numeric(7, 3),
	"co" numeric(8, 3),
	"pollen_tree_count" integer,
	"pollen_grass_count" integer,
	"pollen_weed_count" integer,
	"pollen_overall" integer,
	"pollen_category" varchar,
	"mold_spore_count" integer,
	"mold_category" varchar,
	"active_hazards" jsonb,
	"weather_source" varchar,
	"aqi_source" varchar,
	"pollen_source" varchar,
	"hazard_source" varchar,
	"data_quality_score" numeric(4, 2),
	"missing_fields" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "environmental_forecasts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"generated_at" timestamp NOT NULL,
	"forecast_horizon" varchar NOT NULL,
	"forecast_target_time" timestamp NOT NULL,
	"predicted_risk_score" numeric(5, 2) NOT NULL,
	"predicted_risk_level" varchar NOT NULL,
	"confidence_interval" jsonb,
	"predicted_weather_risk" numeric(5, 2),
	"predicted_air_quality_risk" numeric(5, 2),
	"predicted_allergen_risk" numeric(5, 2),
	"predicted_values" jsonb,
	"model_name" varchar NOT NULL,
	"model_version" varchar NOT NULL,
	"feature_importance" jsonb,
	"actual_risk_score" numeric(5, 2),
	"forecast_error" numeric(5, 2),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "environmental_pipeline_jobs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"job_type" varchar NOT NULL,
	"target_zip_codes" jsonb,
	"target_patient_ids" jsonb,
	"status" varchar NOT NULL,
	"started_at" timestamp,
	"completed_at" timestamp,
	"records_processed" integer DEFAULT 0,
	"records_created" integer DEFAULT 0,
	"records_updated" integer DEFAULT 0,
	"alerts_generated" integer DEFAULT 0,
	"error_message" text,
	"error_stack" text,
	"retry_count" integer DEFAULT 0,
	"max_retries" integer DEFAULT 3,
	"execution_time_ms" integer,
	"trigger_source" varchar,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "genetic_risk_flags" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"flag_name" varchar NOT NULL,
	"flag_type" varchar NOT NULL,
	"value" varchar NOT NULL,
	"risk_level" varchar,
	"clinical_implications" text,
	"affected_medications" jsonb,
	"affected_conditions" jsonb,
	"source" varchar NOT NULL,
	"source_record_id" varchar,
	"testing_provider" varchar,
	"recorded_date" timestamp,
	"auto_generated" boolean DEFAULT true,
	"manual_override" boolean DEFAULT false,
	"overridden_by" varchar,
	"overridden_at" timestamp,
	"doctor_notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "gmail_sync" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"access_token" text,
	"refresh_token" text,
	"token_expiry" timestamp,
	"token_scopes" text[],
	"google_workspace_domain" varchar,
	"google_workspace_baa_confirmed" boolean DEFAULT false,
	"sync_enabled" boolean DEFAULT false,
	"last_sync_at" timestamp,
	"last_sync_status" varchar,
	"last_sync_error" text,
	"total_emails_synced" integer DEFAULT 0,
	"phi_redaction_enabled" boolean DEFAULT true,
	"consent_confirmed" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "gmail_sync_doctor_id_unique" UNIQUE("doctor_id")
);
--> statement-breakpoint
CREATE TABLE "gmail_sync_logs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"sync_id" varchar,
	"action" varchar NOT NULL,
	"status" varchar NOT NULL,
	"emails_fetched" integer DEFAULT 0,
	"phi_detected" boolean DEFAULT false,
	"error" text,
	"error_details" jsonb,
	"ip_address" varchar,
	"user_agent" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "google_calendar_sync" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"access_token" text,
	"refresh_token" text,
	"token_expiry" timestamp,
	"calendar_id" varchar,
	"calendar_name" varchar,
	"sync_enabled" boolean DEFAULT true,
	"sync_direction" varchar DEFAULT 'bidirectional',
	"last_sync_at" timestamp,
	"last_sync_status" varchar,
	"last_sync_error" text,
	"sync_token" text,
	"page_token" text,
	"total_events_synced" integer DEFAULT 0,
	"last_event_synced_at" timestamp,
	"conflict_resolution" varchar DEFAULT 'google_wins',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "google_calendar_sync_doctor_id_unique" UNIQUE("doctor_id")
);
--> statement-breakpoint
CREATE TABLE "google_calendar_sync_logs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"sync_id" varchar,
	"sync_type" varchar NOT NULL,
	"sync_direction" varchar NOT NULL,
	"status" varchar NOT NULL,
	"events_created" integer DEFAULT 0,
	"events_updated" integer DEFAULT 0,
	"events_deleted" integer DEFAULT 0,
	"conflicts_detected" integer DEFAULT 0,
	"error" text,
	"error_details" jsonb,
	"duration_ms" integer,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_ai_recommendations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"habit_id" varchar,
	"recommendation_type" varchar NOT NULL,
	"title" varchar NOT NULL,
	"description" text NOT NULL,
	"based_on_completion_rate" numeric(3, 2),
	"based_on_streak" integer,
	"based_on_mood_trend" varchar,
	"suggested_change" jsonb,
	"confidence" numeric(3, 2),
	"priority" varchar,
	"status" varchar DEFAULT 'pending',
	"user_response" text,
	"expires_at" timestamp,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_ai_triggers" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"habit_id" varchar,
	"trigger_type" varchar NOT NULL,
	"pattern" text NOT NULL,
	"correlated_factor" varchar,
	"correlation_strength" numeric(3, 2),
	"confidence" numeric(3, 2),
	"data_points" integer DEFAULT 0,
	"sample_period_days" integer,
	"acknowledged" boolean DEFAULT false,
	"helpful" boolean,
	"user_notes" text,
	"is_active" boolean DEFAULT true,
	"last_detected_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_buddies" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"buddy_user_id" varchar NOT NULL,
	"status" varchar DEFAULT 'pending',
	"initiated_by" varchar,
	"share_streak" boolean DEFAULT true,
	"share_completions" boolean DEFAULT true,
	"share_mood" boolean DEFAULT false,
	"shared_habit_ids" jsonb,
	"encouragements_sent" integer DEFAULT 0,
	"encouragements_received" integer DEFAULT 0,
	"last_interaction" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_cbt_sessions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"session_type" varchar NOT NULL,
	"title" varchar NOT NULL,
	"current_step" integer DEFAULT 1,
	"total_steps" integer NOT NULL,
	"step_responses" jsonb,
	"completed" boolean DEFAULT false,
	"completed_at" timestamp,
	"pre_session_mood" integer,
	"post_session_mood" integer,
	"helpful_rating" integer,
	"notes" text,
	"related_habit_id" varchar,
	"related_quit_plan_id" varchar,
	"started_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_coach_chats" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"session_id" varchar NOT NULL,
	"role" varchar NOT NULL,
	"content" text NOT NULL,
	"related_habit_id" varchar,
	"related_quit_plan_id" varchar,
	"coach_personality" varchar,
	"response_type" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_cravings_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"quit_plan_id" varchar NOT NULL,
	"user_id" varchar NOT NULL,
	"intensity" integer NOT NULL,
	"duration" integer,
	"trigger" varchar,
	"trigger_details" text,
	"coping_strategy_used" varchar,
	"overcame" boolean DEFAULT false,
	"notes" text,
	"location" varchar,
	"time_of_day" varchar,
	"mood" varchar,
	"occurred_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_encouragements" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"from_user_id" varchar NOT NULL,
	"to_user_id" varchar NOT NULL,
	"message_type" varchar,
	"message" text NOT NULL,
	"prebuilt_message_id" varchar,
	"related_habit_id" varchar,
	"related_achievement" varchar,
	"read" boolean DEFAULT false,
	"read_at" timestamp,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_journals" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"title" varchar,
	"content" text NOT NULL,
	"entry_type" varchar,
	"ai_summary" text,
	"highlights" jsonb,
	"risks" jsonb,
	"recommendations" jsonb,
	"sentiment_trend" varchar,
	"tags" jsonb,
	"mood" varchar,
	"is_weekly_summary" boolean DEFAULT false,
	"week_start_date" timestamp,
	"recorded_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_micro_steps" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"habit_id" varchar NOT NULL,
	"step_order" integer NOT NULL,
	"title" varchar NOT NULL,
	"description" text,
	"estimated_minutes" integer,
	"is_required" boolean DEFAULT true,
	"completion_count" integer DEFAULT 0,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_mood_entries" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"mood_score" integer NOT NULL,
	"mood_label" varchar,
	"energy_level" integer,
	"stress_level" integer,
	"journal_text" text,
	"sentiment_score" numeric(3, 2),
	"extracted_emotions" jsonb,
	"extracted_themes" jsonb,
	"associated_habit_id" varchar,
	"context_tags" jsonb,
	"recorded_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_quit_plans" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"habit_name" varchar NOT NULL,
	"category" varchar,
	"quit_method" varchar,
	"target_quit_date" timestamp,
	"daily_limit" integer,
	"harm_reduction_steps" jsonb,
	"reasons_to_quit" jsonb,
	"money_saved_per_day" numeric(10, 2),
	"start_date" timestamp,
	"days_clean" integer DEFAULT 0,
	"longest_streak" integer DEFAULT 0,
	"total_relapses" integer DEFAULT 0,
	"status" varchar DEFAULT 'active',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_relapse_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"quit_plan_id" varchar NOT NULL,
	"user_id" varchar NOT NULL,
	"severity" varchar,
	"quantity" varchar,
	"trigger" varchar,
	"emotional_state" varchar,
	"what_happened" text,
	"what_learned" text,
	"plan_to_prevent" text,
	"streak_days_lost" integer,
	"occurred_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_reminders" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"habit_id" varchar NOT NULL,
	"user_id" varchar NOT NULL,
	"reminder_type" varchar NOT NULL,
	"scheduled_time" varchar NOT NULL,
	"message" text,
	"adaptive_enabled" boolean DEFAULT true,
	"learned_best_time" varchar,
	"is_active" boolean DEFAULT true,
	"snooze_until" timestamp,
	"last_sent_at" timestamp,
	"times_delivered" integer DEFAULT 0,
	"times_acted_on" integer DEFAULT 0,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_rewards" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"reward_type" varchar DEFAULT 'sunflower',
	"current_level" integer DEFAULT 1,
	"growth_stage" varchar DEFAULT 'seed',
	"total_points" integer DEFAULT 0,
	"streak_bonus" integer DEFAULT 0,
	"completion_points" integer DEFAULT 0,
	"visual_state" jsonb,
	"unlocked_badges" jsonb,
	"unlocked_themes" jsonb,
	"days_active" integer DEFAULT 0,
	"perfect_days" integer DEFAULT 0,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_risk_alerts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"alert_type" varchar NOT NULL,
	"severity" varchar NOT NULL,
	"title" varchar NOT NULL,
	"message" text NOT NULL,
	"risk_score" numeric(3, 2),
	"contributing_factors" jsonb,
	"related_habit_ids" jsonb,
	"related_quit_plan_id" varchar,
	"suggested_actions" jsonb,
	"status" varchar DEFAULT 'active',
	"acknowledged_at" timestamp,
	"predicted_for" timestamp,
	"expires_at" timestamp,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "habit_routines" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"habit_id" varchar NOT NULL,
	"user_id" varchar NOT NULL,
	"scheduled_time" varchar,
	"duration" integer,
	"time_flexibility" varchar,
	"location" varchar,
	"location_details" text,
	"trigger_cue" varchar,
	"stacked_after" varchar,
	"day_of_week" jsonb,
	"is_weekend_only" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "health_section_analytics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"health_section" varchar NOT NULL,
	"analysis_date" timestamp NOT NULL,
	"window_start" timestamp NOT NULL,
	"window_end" timestamp NOT NULL,
	"window_days" integer DEFAULT 7,
	"device_types" jsonb,
	"reading_count" integer DEFAULT 0,
	"ml_prediction" jsonb,
	"deterioration_index" numeric(5, 2),
	"deterioration_trend" varchar,
	"deterioration_change_rate" numeric(5, 2),
	"risk_score" numeric(5, 2),
	"risk_level" varchar,
	"risk_factors" jsonb,
	"trend_direction" varchar,
	"trend_strength" numeric(5, 2),
	"trend_metrics" jsonb,
	"stability_score" numeric(5, 2),
	"variability_index" numeric(5, 2),
	"stability_metrics" jsonb,
	"alerts_generated" jsonb,
	"active_alert_count" integer DEFAULT 0,
	"section_metrics" jsonb,
	"baseline_deviation" numeric(5, 2),
	"baseline_status" varchar,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "healthcare_locations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" varchar NOT NULL,
	"location_type" varchar NOT NULL,
	"address" text,
	"city" varchar,
	"state" varchar,
	"zip_code" varchar,
	"country" varchar DEFAULT 'USA',
	"latitude" numeric,
	"longitude" numeric,
	"phone_number" varchar,
	"is_active" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "infectious_events" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"infection_type" varchar NOT NULL,
	"pathogen" varchar,
	"pathogen_category" varchar,
	"severity" varchar DEFAULT 'moderate' NOT NULL,
	"onset_date" timestamp,
	"resolution_date" timestamp,
	"duration_days" integer,
	"hospitalization" boolean DEFAULT false,
	"icu_admission" boolean DEFAULT false,
	"ventilator_required" boolean DEFAULT false,
	"related_condition_id" varchar,
	"related_condition_code" varchar,
	"related_visit_id" varchar,
	"location_id" varchar,
	"auto_generated" boolean DEFAULT true,
	"manual_override" boolean DEFAULT false,
	"overridden_by" varchar,
	"overridden_at" timestamp,
	"doctor_notes" text,
	"last_etl_processed_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "lysa_clinical_insights" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"doctor_id" varchar,
	"monitoring_assignment_id" varchar,
	"insight_type" varchar NOT NULL,
	"insight_category" varchar NOT NULL,
	"severity" varchar DEFAULT 'info' NOT NULL,
	"title" varchar NOT NULL,
	"summary" text NOT NULL,
	"detailed_analysis" text,
	"evidence_points" jsonb,
	"ai_reasoning" text,
	"confidence_score" numeric(3, 2),
	"suggested_actions" jsonb,
	"related_diagnoses" jsonb,
	"related_medications" jsonb,
	"clinical_guidelines" jsonb,
	"status" varchar DEFAULT 'new',
	"viewed_at" timestamp,
	"acknowledged_at" timestamp,
	"doctor_notes" text,
	"action_taken" text,
	"valid_until" timestamp,
	"superseded_by" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "lysa_monitoring_artifacts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"monitoring_assignment_id" varchar,
	"patient_id" varchar NOT NULL,
	"doctor_id" varchar,
	"artifact_type" varchar NOT NULL,
	"artifact_format" varchar DEFAULT 'markdown' NOT NULL,
	"title" varchar NOT NULL,
	"description" text,
	"content" text,
	"structured_data" jsonb,
	"file_url" varchar,
	"file_size" integer,
	"file_mime_type" varchar,
	"period_start" timestamp,
	"period_end" timestamp,
	"generated_by" varchar DEFAULT 'lysa_ai',
	"ai_model_version" varchar,
	"generation_prompt" text,
	"requires_review" boolean DEFAULT true,
	"reviewed_by" varchar,
	"reviewed_at" timestamp,
	"review_notes" text,
	"approved_for_patient" boolean DEFAULT false,
	"shared_with_patient" boolean DEFAULT false,
	"shared_at" timestamp,
	"version" integer DEFAULT 1,
	"previous_version_id" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "lysa_monitoring_assignments" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"monitoring_level" varchar DEFAULT 'standard' NOT NULL,
	"is_active" boolean DEFAULT true NOT NULL,
	"alert_thresholds" jsonb,
	"check_frequency" varchar DEFAULT 'daily',
	"last_check_at" timestamp,
	"next_scheduled_check" timestamp,
	"auto_generate_summaries" boolean DEFAULT true,
	"summary_frequency" varchar DEFAULT 'daily',
	"last_summary_at" timestamp,
	"enable_alerts" boolean DEFAULT true,
	"alert_channels" jsonb,
	"monitoring_notes" text,
	"focus_areas" jsonb,
	"status" varchar DEFAULT 'active' NOT NULL,
	"paused_at" timestamp,
	"pause_reason" text,
	"completed_at" timestamp,
	"completion_reason" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "lysa_monitoring_events" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"monitoring_assignment_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"doctor_id" varchar,
	"event_type" varchar NOT NULL,
	"event_category" varchar NOT NULL,
	"severity" varchar DEFAULT 'info',
	"title" varchar NOT NULL,
	"description" text,
	"ai_analysis" text,
	"ai_recommendations" jsonb,
	"ai_confidence" numeric(3, 2),
	"source_data_type" varchar,
	"source_data_id" varchar,
	"related_metrics" jsonb,
	"action_required" boolean DEFAULT false,
	"action_taken" varchar,
	"action_taken_by" varchar,
	"action_taken_at" timestamp,
	"notified_doctor" boolean DEFAULT false,
	"notified_patient" boolean DEFAULT false,
	"notification_method" varchar,
	"status" varchar DEFAULT 'new',
	"resolved_at" timestamp,
	"resolved_by" varchar,
	"resolution_notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "medication_change_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"medication_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"change_type" varchar NOT NULL,
	"changed_by" varchar NOT NULL,
	"changed_by_user_id" varchar NOT NULL,
	"old_dosage" varchar,
	"new_dosage" varchar,
	"old_frequency" varchar,
	"new_frequency" varchar,
	"discontinuation_reason" text,
	"replacement_medication_id" varchar,
	"change_reason" text,
	"notes" text,
	"ip_address" varchar,
	"user_agent" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "medication_conflicts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"conflict_group_id" varchar NOT NULL,
	"medication1_id" varchar,
	"medication2_id" varchar,
	"prescription1_id" varchar,
	"prescription2_id" varchar,
	"doctor1_id" varchar NOT NULL,
	"doctor2_id" varchar NOT NULL,
	"specialty1" varchar NOT NULL,
	"specialty2" varchar NOT NULL,
	"conflict_type" varchar NOT NULL,
	"severity" varchar NOT NULL,
	"description" text NOT NULL,
	"detected_reason" text,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"doctor1_response" text,
	"doctor1_responded_at" timestamp,
	"doctor1_action" varchar,
	"doctor2_response" text,
	"doctor2_responded_at" timestamp,
	"doctor2_action" varchar,
	"resolution" varchar,
	"resolution_details" text,
	"resolved_by" varchar,
	"resolved_at" timestamp,
	"doctor1_notified_at" timestamp,
	"doctor2_notified_at" timestamp,
	"patient_notified_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "medication_drug_matches" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"medication_id" varchar NOT NULL,
	"drug_id" varchar NOT NULL,
	"match_source" varchar NOT NULL,
	"confidence_score" numeric NOT NULL,
	"matched_by" varchar,
	"matched_at" timestamp DEFAULT now(),
	"is_active" boolean DEFAULT true,
	"match_metadata" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "mental_health_pattern_analysis" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"response_id" varchar,
	"analysis_type" varchar NOT NULL,
	"patterns" jsonb,
	"symptom_clusters" jsonb,
	"temporal_trends" jsonb,
	"neutral_summary" text,
	"key_observations" jsonb,
	"suggested_actions" jsonb,
	"llm_model" varchar DEFAULT 'gpt-4o',
	"llm_tokens_used" integer,
	"analysis_version" varchar DEFAULT '1.0',
	"analysis_completed_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "mental_health_red_flags" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"session_id" varchar NOT NULL,
	"message_id" varchar,
	"raw_text" text NOT NULL,
	"extracted_json" jsonb NOT NULL,
	"confidence" numeric(3, 2),
	"extraction_model" varchar DEFAULT 'gpt-4o',
	"severity_score" integer,
	"requires_immediate_attention" boolean DEFAULT false,
	"clinician_notified" boolean DEFAULT false,
	"clinician_notified_at" timestamp,
	"reviewed_by" varchar,
	"reviewed_at" timestamp,
	"review_notes" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "mental_health_responses" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"questionnaire_type" varchar NOT NULL,
	"questionnaire_version" varchar DEFAULT '1.0',
	"responses" jsonb NOT NULL,
	"total_score" integer,
	"max_score" integer,
	"severity_level" varchar,
	"cluster_scores" jsonb,
	"crisis_detected" boolean DEFAULT false,
	"crisis_question_ids" jsonb,
	"crisis_responses" jsonb,
	"completed_at" timestamp DEFAULT now(),
	"duration_seconds" integer,
	"allow_storage" boolean DEFAULT true,
	"allow_clinical_sharing" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ml_extraction_audit_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"request_id" varchar NOT NULL,
	"requester_id" varchar NOT NULL,
	"purpose" text NOT NULL,
	"policy_id" varchar NOT NULL,
	"n_patients_consented" integer NOT NULL,
	"n_records_extracted" integer NOT NULL,
	"differential_privacy_applied" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ml_models" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"model_name" varchar NOT NULL,
	"model_type" varchar NOT NULL,
	"version" varchar NOT NULL,
	"status" varchar DEFAULT 'training' NOT NULL,
	"is_active" boolean DEFAULT false,
	"training_config" jsonb,
	"training_data_sources" jsonb,
	"metrics" jsonb,
	"model_path" varchar,
	"model_format" varchar DEFAULT 'onnx',
	"model_size_bytes" integer,
	"feature_names" jsonb,
	"feature_importance" jsonb,
	"trained_by" varchar,
	"training_started_at" timestamp,
	"training_completed_at" timestamp,
	"training_duration_seconds" integer,
	"deployed_at" timestamp,
	"deprecated_at" timestamp,
	"deprecation_reason" text,
	"previous_version_id" varchar,
	"improvement_over_previous" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ml_training_audit_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"event_type" varchar NOT NULL,
	"event_category" varchar NOT NULL,
	"actor_id" varchar,
	"actor_type" varchar NOT NULL,
	"resource_type" varchar NOT NULL,
	"resource_id" varchar,
	"patient_id_hash" varchar,
	"phi_accessed" boolean DEFAULT false,
	"phi_categories" jsonb,
	"event_details" jsonb,
	"previous_state" jsonb,
	"new_state" jsonb,
	"ip_address" varchar,
	"user_agent" text,
	"session_id" varchar,
	"success" boolean DEFAULT true,
	"error_message" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ml_training_consent" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"consent_enabled" boolean DEFAULT false NOT NULL,
	"data_types" jsonb DEFAULT '{"vitals":true,"symptoms":true,"medications":false,"mentalHealth":false,"behavioralData":true,"wearableData":true,"labResults":false,"imagingData":false}'::jsonb,
	"anonymization_level" varchar DEFAULT 'full',
	"consent_version" varchar DEFAULT '1.0',
	"consent_signed_at" timestamp,
	"consent_withdrawn_at" timestamp,
	"withdrawal_reason" text,
	"requested_data_deletion" boolean DEFAULT false,
	"data_deletion_requested_at" timestamp,
	"data_deletion_completed_at" timestamp,
	"last_modified_by" varchar,
	"ip_address" varchar,
	"user_agent" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ml_training_contributions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id_hash" varchar NOT NULL,
	"consent_id" varchar NOT NULL,
	"training_job_id" varchar,
	"model_id" varchar,
	"data_types_contributed" jsonb,
	"record_count" integer NOT NULL,
	"date_range_start" timestamp,
	"date_range_end" timestamp,
	"anonymization_level" varchar NOT NULL,
	"status" varchar DEFAULT 'included',
	"contributed_at" timestamp DEFAULT now(),
	"withdrawn_at" timestamp,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ml_training_jobs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"job_name" varchar NOT NULL,
	"model_name" varchar NOT NULL,
	"target_version" varchar NOT NULL,
	"status" varchar DEFAULT 'queued' NOT NULL,
	"priority" integer DEFAULT 5,
	"data_sources" jsonb,
	"training_config" jsonb,
	"current_phase" varchar,
	"progress_percent" integer DEFAULT 0,
	"progress_message" text,
	"resource_usage" jsonb,
	"queued_at" timestamp DEFAULT now(),
	"started_at" timestamp,
	"completed_at" timestamp,
	"estimated_completion_at" timestamp,
	"result_model_id" varchar,
	"error_message" text,
	"error_details" jsonb,
	"notify_on_completion" boolean DEFAULT true,
	"notification_email" varchar,
	"initiated_by" varchar,
	"logs_path" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "occupational_exposures" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"occupation_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"exposure_type" varchar NOT NULL,
	"exposure_agent" varchar,
	"exposure_level" varchar DEFAULT 'medium' NOT NULL,
	"exposure_frequency" varchar,
	"exposure_duration_years" numeric,
	"protective_equipment_used" boolean DEFAULT false,
	"protective_equipment_details" text,
	"health_impact_notes" text,
	"auto_generated" boolean DEFAULT true,
	"manual_override" boolean DEFAULT false,
	"modified_by" varchar,
	"modified_at" timestamp,
	"doctor_notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "paintrack_sessions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"module" varchar NOT NULL,
	"joint" varchar NOT NULL,
	"laterality" varchar,
	"front_video_url" varchar,
	"joint_video_url" varchar,
	"patient_vas" integer,
	"patient_notes" text,
	"medication_taken" boolean DEFAULT false,
	"medication_details" text,
	"recording_duration" integer,
	"device_type" varchar,
	"dual_camera_supported" boolean DEFAULT false,
	"status" varchar DEFAULT 'pending',
	"processing_error" text,
	"video_quality" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "passive_metrics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"date" timestamp NOT NULL,
	"steps" integer,
	"active_minutes" integer,
	"calories_burned" integer,
	"distance_meters" integer,
	"hr_mean" integer,
	"hr_min" integer,
	"hr_max" integer,
	"hrv" integer,
	"resting_hr" integer,
	"sleep_minutes" integer,
	"deep_sleep_minutes" integer,
	"rem_sleep_minutes" integer,
	"light_sleep_minutes" integer,
	"awake_minutes" integer,
	"sleep_score" integer,
	"respiratory_rate" integer,
	"spo2_mean" integer,
	"spo2_min" integer,
	"stress_score" integer,
	"recovery_score" integer,
	"device_meta" jsonb,
	"synced_at" timestamp DEFAULT now(),
	"data_quality" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_conditions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"condition_code" varchar NOT NULL,
	"condition_name" varchar NOT NULL,
	"condition_category" varchar,
	"status" varchar DEFAULT 'active',
	"severity" varchar,
	"onset_date" timestamp,
	"diagnosis_date" timestamp,
	"resolution_date" timestamp,
	"diagnosed_by" varchar,
	"source_type" varchar DEFAULT 'ehr',
	"source_record_id" varchar,
	"notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_consent_requests" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"doctor_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"request_message" text,
	"responded_at" timestamp,
	"response_message" text,
	"expires_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_environment_profiles" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"zip_code" varchar(10) NOT NULL,
	"city" varchar,
	"state" varchar(2),
	"timezone" varchar DEFAULT 'America/New_York',
	"chronic_conditions" jsonb,
	"allergies" jsonb,
	"alerts_enabled" boolean DEFAULT true,
	"alert_thresholds" jsonb,
	"push_notifications" boolean DEFAULT true,
	"sms_notifications" boolean DEFAULT false,
	"email_digest" boolean DEFAULT true,
	"digest_frequency" varchar DEFAULT 'daily',
	"correlation_consent_given" boolean DEFAULT false,
	"correlation_consent_at" timestamp,
	"is_active" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_environment_risk_scores" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"snapshot_id" varchar,
	"computed_at" timestamp NOT NULL,
	"composite_risk_score" numeric(5, 2) NOT NULL,
	"risk_level" varchar NOT NULL,
	"weather_risk_score" numeric(5, 2),
	"air_quality_risk_score" numeric(5, 2),
	"allergen_risk_score" numeric(5, 2),
	"hazard_risk_score" numeric(5, 2),
	"trend_24hr" numeric(6, 3),
	"trend_48hr" numeric(6, 3),
	"trend_72hr" numeric(6, 3),
	"volatility_score" numeric(5, 2),
	"factor_contributions" jsonb,
	"top_risk_factors" jsonb,
	"scoring_version" varchar DEFAULT '1.0',
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_immunizations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"vaccine_code" varchar,
	"vaccine_name" varchar NOT NULL,
	"vaccine_manufacturer" varchar,
	"lot_number" varchar,
	"dose_number" integer,
	"series_name" varchar,
	"series_complete" boolean DEFAULT false,
	"administration_date" timestamp NOT NULL,
	"administration_route" varchar,
	"administration_site" varchar,
	"location_id" varchar,
	"administered_by" varchar,
	"status" varchar DEFAULT 'completed',
	"adverse_reaction" boolean DEFAULT false,
	"reaction_details" text,
	"source_type" varchar DEFAULT 'ehr',
	"source_record_id" varchar,
	"auto_generated" boolean DEFAULT false,
	"manual_override" boolean DEFAULT false,
	"overridden_by" varchar,
	"overridden_at" timestamp,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_occupations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"job_title" varchar NOT NULL,
	"industry" varchar,
	"employer" varchar,
	"physical_demand_level" varchar,
	"shift_work" boolean DEFAULT false,
	"night_shift" boolean DEFAULT false,
	"hours_per_week" integer,
	"start_date" timestamp,
	"end_date" timestamp,
	"is_current" boolean DEFAULT true,
	"status" varchar DEFAULT 'active',
	"auto_enriched" boolean DEFAULT false,
	"enriched_at" timestamp,
	"created_by" varchar,
	"modified_by" varchar,
	"modified_at" timestamp,
	"doctor_notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_trigger_weights" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"factor_type" varchar NOT NULL,
	"personalized_weight" numeric(5, 4) NOT NULL,
	"confidence_score" numeric(5, 4),
	"source" varchar NOT NULL,
	"correlation_coefficient" numeric(5, 4),
	"p_value" numeric(8, 6),
	"sample_size" integer,
	"last_updated_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "patient_visits" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"visit_type" varchar NOT NULL,
	"location_id" varchar,
	"facility_name" varchar,
	"admission_date" timestamp NOT NULL,
	"discharge_date" timestamp,
	"length_of_stay" integer,
	"is_hospitalization" boolean DEFAULT false,
	"icu_admission" boolean DEFAULT false,
	"icu_days" integer,
	"ventilator_required" boolean DEFAULT false,
	"chief_complaint" text,
	"primary_diagnosis_code" varchar,
	"secondary_diagnoses_codes" jsonb,
	"discharge_disposition" varchar,
	"source_type" varchar DEFAULT 'ehr',
	"source_record_id" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "prescriptions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"doctor_id" varchar NOT NULL,
	"drug_id" varchar,
	"rxcui" varchar,
	"medication_name" varchar NOT NULL,
	"dosage" varchar NOT NULL,
	"frequency" varchar NOT NULL,
	"dosage_instructions" text,
	"quantity" integer,
	"refills" integer DEFAULT 0,
	"start_date" timestamp DEFAULT now(),
	"expiration_date" timestamp,
	"is_continuous" boolean DEFAULT false,
	"duration_days" integer,
	"intended_start_date" timestamp,
	"specialty" varchar,
	"supersedes" varchar,
	"superseded_by" varchar,
	"superseded_at" timestamp,
	"conflict_group_id" varchar,
	"has_conflict" boolean DEFAULT false,
	"conflict_detected_at" timestamp,
	"conflict_resolved_at" timestamp,
	"status" varchar DEFAULT 'sent' NOT NULL,
	"acknowledged_at" timestamp,
	"acknowledged_by" varchar,
	"document_id" varchar,
	"medication_id" varchar,
	"notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "protocol_version_audit_log" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"protocol_id" varchar NOT NULL,
	"action" varchar NOT NULL,
	"old_version" varchar,
	"new_version" varchar,
	"user_id" varchar NOT NULL,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "public_dataset_registry" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"dataset_name" varchar NOT NULL,
	"source" varchar NOT NULL,
	"version" varchar,
	"requires_credentials" boolean DEFAULT true,
	"credentials_configured" boolean DEFAULT false,
	"access_url" varchar,
	"documentation_url" varchar,
	"description" text,
	"record_count" integer,
	"patient_count" integer,
	"date_range" jsonb,
	"data_types" jsonb,
	"total_size_gb" numeric(10, 2),
	"file_formats" jsonb,
	"download_status" varchar DEFAULT 'not_started',
	"download_progress" integer DEFAULT 0,
	"local_path" varchar,
	"preprocessed_at" timestamp,
	"preprocessing_config" jsonb,
	"used_in_models" jsonb,
	"last_used_at" timestamp,
	"license" varchar,
	"citation_required" boolean DEFAULT true,
	"citation" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "public_dataset_registry_dataset_name_unique" UNIQUE("dataset_name")
);
--> statement-breakpoint
CREATE TABLE "research_alerts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar,
	"study_id" varchar,
	"alert_type" varchar NOT NULL,
	"severity" varchar NOT NULL,
	"risk_score" numeric(5, 2),
	"title" varchar NOT NULL,
	"message" text,
	"details_json" jsonb,
	"status" varchar DEFAULT 'new',
	"acknowledged_at" timestamp,
	"acknowledged_by" varchar,
	"resolved_at" timestamp,
	"resolved_by" varchar,
	"resolution" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_analysis_reports" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" varchar NOT NULL,
	"study_id" varchar,
	"cohort_id" varchar,
	"snapshot_id" varchar,
	"analysis_type" varchar NOT NULL,
	"analysis_spec_json" jsonb,
	"results_json" jsonb,
	"generated_text" text,
	"abstract" text,
	"methods" text,
	"results" text,
	"discussion" text,
	"limitations" text,
	"status" varchar DEFAULT 'draft',
	"publish_ready" boolean DEFAULT false,
	"published_at" timestamp,
	"created_by" varchar NOT NULL,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_audit_logs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"action_type" varchar NOT NULL,
	"object_type" varchar NOT NULL,
	"object_id" varchar NOT NULL,
	"metadata_json" jsonb,
	"success" boolean DEFAULT true,
	"error_message" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_cohorts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"project_id" varchar,
	"name" varchar NOT NULL,
	"description" text,
	"definition_json" jsonb NOT NULL,
	"cached_patient_count" integer DEFAULT 0,
	"cached_stats" jsonb,
	"status" varchar DEFAULT 'active',
	"created_by" varchar NOT NULL,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_data_consent" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"consent_enabled" boolean DEFAULT false NOT NULL,
	"data_type_permissions" jsonb DEFAULT '{"dailyFollowups":false,"healthAlerts":false,"deteriorationIndex":false,"mlPredictions":false,"environmentalRisk":false,"medications":false,"vitals":false,"immuneMarkers":false,"behavioralData":false,"mentalHealth":false,"wearableData":false,"labResults":false,"conditions":false,"demographics":false,"painTracking":false,"symptomJournal":false}'::jsonb,
	"anonymization_level" varchar DEFAULT 'full',
	"allow_reidentification" boolean DEFAULT false,
	"data_retention_years" integer DEFAULT 10,
	"allow_data_export" boolean DEFAULT false,
	"allow_study_enrollment" boolean DEFAULT true,
	"allow_contact_for_studies" boolean DEFAULT false,
	"preferred_contact_method" varchar DEFAULT 'email',
	"consent_version" varchar DEFAULT '1.0',
	"consent_signed_at" timestamp,
	"consent_updated_at" timestamp,
	"consent_withdrawn_at" timestamp,
	"withdrawal_reason" text,
	"legal_basis" varchar DEFAULT 'consent',
	"ethics_approval_ref" varchar,
	"last_modified_by" varchar,
	"ip_address" varchar,
	"user_agent" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_data_snapshots" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"description" varchar,
	"study_id" varchar,
	"cohort_id" varchar,
	"patient_count" integer,
	"measurement_count" integer,
	"followup_count" integer,
	"data_hash" varchar,
	"created_by" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_environmental_exposures" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"location_id" varchar NOT NULL,
	"date" timestamp NOT NULL,
	"pm25" numeric(8, 3),
	"pm10" numeric(8, 3),
	"ozone" numeric(8, 3),
	"air_quality_index" integer,
	"air_quality_category" varchar,
	"temperature_high" numeric(5, 2),
	"temperature_low" numeric(5, 2),
	"temperature_avg" numeric(5, 2),
	"humidity" integer,
	"barometric_pressure" numeric(7, 2),
	"precipitation" numeric(6, 2),
	"uv_index" integer,
	"pollen_index" integer,
	"pollen_tree" integer,
	"pollen_grass" integer,
	"pollen_ragweed" integer,
	"mold_index" integer,
	"wildfires_nearby" boolean DEFAULT false,
	"dust_storm" boolean DEFAULT false,
	"data_source" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_immune_markers" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"study_id" varchar,
	"visit_id" varchar,
	"marker_name" varchar NOT NULL,
	"value" numeric(15, 5) NOT NULL,
	"units" varchar NOT NULL,
	"reference_range_low" numeric(15, 5),
	"reference_range_high" numeric(15, 5),
	"interpretation" varchar,
	"source" varchar DEFAULT 'lab',
	"lab_name" varchar,
	"specimen_type" varchar,
	"collection_time" timestamp NOT NULL,
	"result_time" timestamp,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_locations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" varchar NOT NULL,
	"region" varchar,
	"city" varchar,
	"state" varchar,
	"country" varchar DEFAULT 'USA',
	"zip_code" varchar,
	"latitude" numeric(10, 7),
	"longitude" numeric(10, 7),
	"population" integer,
	"urban_rural_classification" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_measurements" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"visit_id" varchar,
	"patient_id" varchar NOT NULL,
	"study_id" varchar,
	"name" varchar NOT NULL,
	"value" varchar NOT NULL,
	"value_numeric" numeric(15, 5),
	"units" varchar,
	"category" varchar,
	"recorded_at" timestamp DEFAULT now() NOT NULL,
	"recorded_by" varchar,
	"source" varchar DEFAULT 'manual',
	"is_valid" boolean DEFAULT true,
	"validation_notes" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_patient_locations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"location_id" varchar NOT NULL,
	"start_date" timestamp NOT NULL,
	"end_date" timestamp,
	"location_type" varchar DEFAULT 'residence',
	"is_primary" boolean DEFAULT true,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_projects" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"owner_id" varchar NOT NULL,
	"name" varchar NOT NULL,
	"description" text,
	"project_type" varchar DEFAULT 'personal',
	"status" varchar DEFAULT 'active',
	"collaborator_ids" jsonb DEFAULT '[]'::jsonb,
	"is_public" boolean DEFAULT false,
	"allow_data_sharing" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_protocols" (
	"id" varchar PRIMARY KEY NOT NULL,
	"title" varchar NOT NULL,
	"description" text,
	"principal_investigator" varchar NOT NULL,
	"status" varchar DEFAULT 'draft' NOT NULL,
	"analysis_spec" jsonb,
	"data_snapshot_id" varchar,
	"version" varchar DEFAULT '1.0.0',
	"irb_number" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_studies" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"project_id" varchar,
	"title" varchar NOT NULL,
	"description" text,
	"status" varchar DEFAULT 'planning',
	"start_date" timestamp,
	"end_date" timestamp,
	"planned_duration" integer,
	"target_sample_size" integer,
	"current_enrollment" integer DEFAULT 0,
	"inclusion_criteria" text,
	"exclusion_criteria" text,
	"inclusion_criteria_json" jsonb,
	"exclusion_criteria_json" jsonb,
	"arms_json" jsonb DEFAULT '[]'::jsonb,
	"visit_schedule_json" jsonb DEFAULT '[]'::jsonb,
	"auto_reanalysis" boolean DEFAULT false,
	"reanalysis_frequency" varchar DEFAULT 'weekly',
	"last_reanalysis_at" timestamp,
	"analysis_spec_json" jsonb,
	"owner_user_id" varchar NOT NULL,
	"cohort_id" varchar,
	"ethics_approval_number" varchar,
	"ethics_approval_date" timestamp,
	"data_protection_assessment" boolean DEFAULT false,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_visits" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"enrollment_id" varchar NOT NULL,
	"study_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"visit_type" varchar NOT NULL,
	"visit_number" integer,
	"scheduled_date" timestamp,
	"window_start_date" timestamp,
	"window_end_date" timestamp,
	"actual_date" timestamp,
	"visit_status" varchar DEFAULT 'scheduled',
	"assessments_completed" jsonb DEFAULT '[]'::jsonb,
	"notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "risk_exposures_etl_jobs" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"job_type" varchar NOT NULL,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"records_processed" integer DEFAULT 0,
	"records_created" integer DEFAULT 0,
	"records_updated" integer DEFAULT 0,
	"records_skipped" integer DEFAULT 0,
	"started_at" timestamp,
	"completed_at" timestamp,
	"execution_time_ms" integer,
	"error_message" text,
	"error_stack" text,
	"last_processed_id" varchar,
	"last_processed_at" timestamp,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "robustness_reports" (
	"id" varchar PRIMARY KEY NOT NULL,
	"protocol_id" varchar,
	"overall_status" varchar NOT NULL,
	"report_json" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "sentiment_analysis" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"source_type" varchar NOT NULL,
	"source_id" varchar,
	"text_content" text NOT NULL,
	"analyzed_at" timestamp NOT NULL,
	"sentiment_polarity" numeric(5, 3),
	"sentiment_label" varchar,
	"sentiment_confidence" numeric(5, 3),
	"message_length_chars" integer,
	"word_count" integer,
	"lexical_complexity" numeric(5, 3),
	"negativity_ratio" numeric(5, 3),
	"stress_keyword_count" integer DEFAULT 0,
	"stress_keywords" jsonb,
	"help_seeking_detected" boolean DEFAULT false,
	"help_seeking_phrases" jsonb,
	"hesitation_count" integer DEFAULT 0,
	"hesitation_markers" jsonb,
	"model_version" varchar DEFAULT 'distilbert-base-uncased-finetuned-sst-2-english',
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "session_metrics" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_id" varchar NOT NULL,
	"joint_metrics" jsonb,
	"facial_metrics" jsonb,
	"anomaly_score" numeric(5, 3),
	"baseline_deviation" numeric(5, 3),
	"correlation_score" numeric(5, 3),
	"model_versions" jsonb,
	"processed_at" timestamp DEFAULT now(),
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "study_enrollments" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"study_id" varchar NOT NULL,
	"patient_id" varchar NOT NULL,
	"arm" varchar,
	"consent_status" varchar DEFAULT 'pending',
	"consent_date" timestamp,
	"withdrawal_date" timestamp,
	"withdrawal_reason" text,
	"enrollment_date" timestamp DEFAULT now(),
	"screening_date" timestamp,
	"status" varchar DEFAULT 'active',
	"notes" text,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "symptom_checkins" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"timestamp" timestamp DEFAULT now() NOT NULL,
	"pain_level" integer,
	"fatigue_level" integer,
	"breathlessness_level" integer,
	"sleep_quality" integer,
	"mood" varchar,
	"mobility_score" integer,
	"medications_taken" boolean,
	"triggers" text[],
	"symptoms" text[],
	"note" text,
	"voice_note_url" varchar,
	"voice_note_duration" integer,
	"source" varchar DEFAULT 'app' NOT NULL,
	"session_id" varchar,
	"device_type" varchar,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "symptom_environment_correlations" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"symptom_type" varchar NOT NULL,
	"symptom_severity_metric" varchar NOT NULL,
	"environmental_factor" varchar NOT NULL,
	"correlation_type" varchar NOT NULL,
	"correlation_coefficient" numeric(6, 4) NOT NULL,
	"p_value" numeric(10, 8),
	"is_statistically_significant" boolean DEFAULT false,
	"optimal_lag" integer,
	"lag_correlation" numeric(6, 4),
	"sample_size" integer NOT NULL,
	"data_window_days" integer,
	"relationship_strength" varchar,
	"relationship_direction" varchar,
	"interpretation" text,
	"confidence_score" numeric(5, 4),
	"data_quality_score" numeric(5, 4),
	"last_analyzed_at" timestamp NOT NULL,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "tool_executions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"agent_id" varchar NOT NULL,
	"user_id" varchar NOT NULL,
	"conversation_id" varchar,
	"message_id" varchar,
	"tool_name" varchar NOT NULL,
	"tool_version" integer DEFAULT 1,
	"input_parameters" jsonb NOT NULL,
	"output_result" jsonb,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"error_message" text,
	"error_code" varchar,
	"execution_time_ms" integer,
	"started_at" timestamp,
	"completed_at" timestamp,
	"patient_id" varchar,
	"doctor_id" varchar,
	"phi_accessed" boolean DEFAULT false,
	"phi_categories" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "trend_reports" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"period_start" timestamp NOT NULL,
	"period_end" timestamp NOT NULL,
	"report_type" varchar NOT NULL,
	"aggregated_metrics" jsonb,
	"anomalies" jsonb,
	"correlations" jsonb,
	"clinician_summary" text,
	"generated_by" varchar DEFAULT 'ml_trend_engine',
	"data_points_analyzed" integer,
	"confidence_score" numeric(3, 2),
	"generated_at" timestamp DEFAULT now(),
	"reviewed_by_doctor" boolean DEFAULT false,
	"reviewed_by" varchar,
	"reviewed_at" timestamp,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "user_presence" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"is_online" boolean DEFAULT false,
	"last_seen_at" timestamp,
	"active_connections" integer DEFAULT 0,
	"connection_ids" jsonb,
	"current_activity" varchar,
	"current_conversation_id" varchar,
	"last_device_info" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now(),
	CONSTRAINT "user_presence_user_id_unique" UNIQUE("user_id")
);
--> statement-breakpoint
CREATE TABLE "vendor_accounts" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"vendor_id" varchar NOT NULL,
	"vendor_name" varchar NOT NULL,
	"access_token" text,
	"refresh_token" text,
	"token_type" varchar DEFAULT 'Bearer',
	"token_expires_at" timestamp,
	"token_scope" text,
	"vendor_user_id" varchar,
	"vendor_username" varchar,
	"connection_status" varchar DEFAULT 'pending',
	"last_auth_at" timestamp,
	"last_refresh_at" timestamp,
	"last_refresh_error" text,
	"refresh_attempts" integer DEFAULT 0,
	"auto_sync" boolean DEFAULT true,
	"sync_frequency" varchar DEFAULT 'hourly',
	"last_sync_at" timestamp,
	"last_sync_status" varchar,
	"last_sync_error" text,
	"synced_data_types" jsonb,
	"webhook_id" varchar,
	"webhook_secret" text,
	"webhook_active" boolean DEFAULT false,
	"consent_granted_at" timestamp,
	"consent_version" varchar,
	"consent_data_types" jsonb,
	"metadata" jsonb,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "video_exam_segments" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_id" varchar NOT NULL,
	"exam_type" varchar NOT NULL,
	"sequence_order" integer NOT NULL,
	"skipped" boolean DEFAULT false,
	"prep_duration_seconds" integer DEFAULT 30,
	"capture_started_at" timestamp,
	"capture_ended_at" timestamp,
	"duration_seconds" integer,
	"s3_key" varchar,
	"s3_bucket" varchar,
	"kms_key_id" varchar,
	"file_size_bytes" integer,
	"analysis_id" varchar,
	"status" varchar DEFAULT 'pending' NOT NULL,
	"custom_location" text,
	"custom_description" text,
	"uploaded_by" varchar,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "video_exam_sessions" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"patient_id" varchar NOT NULL,
	"started_at" timestamp DEFAULT now(),
	"completed_at" timestamp,
	"combined_s3_key" varchar,
	"combined_s3_bucket" varchar,
	"combined_kms_key_id" varchar,
	"combined_file_size_bytes" integer,
	"combined_analysis_id" varchar,
	"status" varchar DEFAULT 'in_progress' NOT NULL,
	"total_segments" integer DEFAULT 0,
	"completed_segments" integer DEFAULT 0,
	"skipped_segments" integer DEFAULT 0,
	"total_duration_seconds" integer DEFAULT 0,
	"created_at" timestamp DEFAULT now(),
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
ALTER TABLE "appointments" ADD COLUMN "booked_by_method" varchar DEFAULT 'online';--> statement-breakpoint
ALTER TABLE "chat_sessions" ADD COLUMN "context_patient_id" varchar;--> statement-breakpoint
ALTER TABLE "drugs" ADD COLUMN "rxcui" varchar;--> statement-breakpoint
ALTER TABLE "drugs" ADD COLUMN "data_source" varchar DEFAULT 'rxnorm';--> statement-breakpoint
ALTER TABLE "drugs" ADD COLUMN "data_version" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "drug_id" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "rxcui" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "source" varchar DEFAULT 'manual' NOT NULL;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "source_document_id" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "source_prescription_id" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "status" varchar DEFAULT 'active' NOT NULL;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "confirmed_at" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "confirmed_by" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "added_by" varchar DEFAULT 'patient' NOT NULL;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "auto_detected" boolean DEFAULT false;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "is_continuous" boolean DEFAULT false;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "intended_start_date" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "actual_start_date" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "duration_days" integer;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "computed_end_date" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "specialty" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "prescribing_doctor_id" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "superseded_by" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "superseded_at" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "supersession_reason" text;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "conflict_group_id" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "conflict_status" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "conflict_detected_at" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "conflict_resolved_at" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "conflict_resolution" text;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "reminders_enabled" boolean DEFAULT true;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "reminder_offsets" jsonb;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "last_reminder_sent_at" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "adherence_log" jsonb;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "discontinued_at" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "discontinued_by" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "discontinuation_reason" text;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "replacement_medication_id" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "archived_at" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "archive_document_id" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "archive_reason" varchar;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "lifecycle_stage" varchar DEFAULT 'active';--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "last_lifecycle_update" timestamp;--> statement-breakpoint
ALTER TABLE "medications" ADD COLUMN "updated_at" timestamp DEFAULT now();--> statement-breakpoint
ALTER TABLE "patient_profiles" ADD COLUMN "followup_patient_id" varchar;--> statement-breakpoint
ALTER TABLE "agent_audit_logs" ADD CONSTRAINT "agent_audit_logs_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_audit_logs" ADD CONSTRAINT "agent_audit_logs_conversation_id_agent_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."agent_conversations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_audit_logs" ADD CONSTRAINT "agent_audit_logs_message_id_agent_messages_id_fk" FOREIGN KEY ("message_id") REFERENCES "public"."agent_messages"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_conversations" ADD CONSTRAINT "agent_conversations_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_conversations" ADD CONSTRAINT "agent_conversations_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_memory" ADD CONSTRAINT "agent_memory_agent_id_agents_id_fk" FOREIGN KEY ("agent_id") REFERENCES "public"."agents"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_memory" ADD CONSTRAINT "agent_memory_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_memory" ADD CONSTRAINT "agent_memory_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_memory" ADD CONSTRAINT "agent_memory_conversation_id_agent_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."agent_conversations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_messages" ADD CONSTRAINT "agent_messages_conversation_id_agent_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."agent_conversations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_messages" ADD CONSTRAINT "agent_messages_approved_by_users_id_fk" FOREIGN KEY ("approved_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_tasks" ADD CONSTRAINT "agent_tasks_agent_id_agents_id_fk" FOREIGN KEY ("agent_id") REFERENCES "public"."agents"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_tasks" ADD CONSTRAINT "agent_tasks_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_tasks" ADD CONSTRAINT "agent_tasks_conversation_id_agent_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."agent_conversations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_tasks" ADD CONSTRAINT "agent_tasks_message_id_agent_messages_id_fk" FOREIGN KEY ("message_id") REFERENCES "public"."agent_messages"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_tool_assignments" ADD CONSTRAINT "agent_tool_assignments_agent_id_agents_id_fk" FOREIGN KEY ("agent_id") REFERENCES "public"."agents"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "agent_tool_assignments" ADD CONSTRAINT "agent_tool_assignments_tool_id_agent_tools_id_fk" FOREIGN KEY ("tool_id") REFERENCES "public"."agent_tools"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_engagement_metrics" ADD CONSTRAINT "ai_engagement_metrics_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_health_alerts" ADD CONSTRAINT "ai_health_alerts_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_health_alerts" ADD CONSTRAINT "ai_health_alerts_acknowledged_by_users_id_fk" FOREIGN KEY ("acknowledged_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_health_alerts" ADD CONSTRAINT "ai_health_alerts_dismissed_by_users_id_fk" FOREIGN KEY ("dismissed_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_health_alerts" ADD CONSTRAINT "ai_health_alerts_clinician_id_users_id_fk" FOREIGN KEY ("clinician_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_qol_metrics" ADD CONSTRAINT "ai_qol_metrics_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ai_trend_metrics" ADD CONSTRAINT "ai_trend_metrics_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "analysis_jobs" ADD CONSTRAINT "analysis_jobs_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "analysis_jobs" ADD CONSTRAINT "analysis_jobs_cohort_id_research_cohorts_id_fk" FOREIGN KEY ("cohort_id") REFERENCES "public"."research_cohorts"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "analysis_jobs" ADD CONSTRAINT "analysis_jobs_report_id_ai_research_reports_id_fk" FOREIGN KEY ("report_id") REFERENCES "public"."ai_research_reports"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "analysis_jobs" ADD CONSTRAINT "analysis_jobs_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "approval_queue" ADD CONSTRAINT "approval_queue_approver_id_users_id_fk" FOREIGN KEY ("approver_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "approval_queue" ADD CONSTRAINT "approval_queue_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "approval_queue" ADD CONSTRAINT "approval_queue_conversation_id_agent_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."agent_conversations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "approval_queue" ADD CONSTRAINT "approval_queue_message_id_agent_messages_id_fk" FOREIGN KEY ("message_id") REFERENCES "public"."agent_messages"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "approval_queue" ADD CONSTRAINT "approval_queue_tool_execution_id_tool_executions_id_fk" FOREIGN KEY ("tool_execution_id") REFERENCES "public"."tool_executions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "approval_queue" ADD CONSTRAINT "approval_queue_decision_by_users_id_fk" FOREIGN KEY ("decision_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "approval_queue" ADD CONSTRAINT "approval_queue_escalated_to_users_id_fk" FOREIGN KEY ("escalated_to") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "behavior_alerts" ADD CONSTRAINT "behavior_alerts_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "behavior_alerts" ADD CONSTRAINT "behavior_alerts_source_risk_score_id_behavior_risk_scores_id_fk" FOREIGN KEY ("source_risk_score_id") REFERENCES "public"."behavior_risk_scores"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "behavior_alerts" ADD CONSTRAINT "behavior_alerts_source_trend_id_deterioration_trends_id_fk" FOREIGN KEY ("source_trend_id") REFERENCES "public"."deterioration_trends"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "behavior_alerts" ADD CONSTRAINT "behavior_alerts_acknowledged_by_users_id_fk" FOREIGN KEY ("acknowledged_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "behavior_checkins" ADD CONSTRAINT "behavior_checkins_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "behavior_metrics" ADD CONSTRAINT "behavior_metrics_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "behavior_risk_scores" ADD CONSTRAINT "behavior_risk_scores_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "chat_symptoms" ADD CONSTRAINT "chat_symptoms_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "chat_symptoms" ADD CONSTRAINT "chat_symptoms_session_id_chat_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."chat_sessions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "chat_symptoms" ADD CONSTRAINT "chat_symptoms_message_id_chat_messages_id_fk" FOREIGN KEY ("message_id") REFERENCES "public"."chat_messages"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "chat_symptoms" ADD CONSTRAINT "chat_symptoms_symptom_checkin_id_symptom_checkins_id_fk" FOREIGN KEY ("symptom_checkin_id") REFERENCES "public"."symptom_checkins"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "clinician_notes" ADD CONSTRAINT "clinician_notes_session_id_paintrack_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."paintrack_sessions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "clinician_notes" ADD CONSTRAINT "clinician_notes_clinician_id_users_id_fk" FOREIGN KEY ("clinician_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "clinician_workload_metrics" ADD CONSTRAINT "clinician_workload_metrics_clinician_id_users_id_fk" FOREIGN KEY ("clinician_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "cognitive_tests" ADD CONSTRAINT "cognitive_tests_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_followup_assignments" ADD CONSTRAINT "daily_followup_assignments_template_id_daily_followup_templates_id_fk" FOREIGN KEY ("template_id") REFERENCES "public"."daily_followup_templates"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_followup_assignments" ADD CONSTRAINT "daily_followup_assignments_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_followup_assignments" ADD CONSTRAINT "daily_followup_assignments_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_followup_assignments" ADD CONSTRAINT "daily_followup_assignments_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_followup_responses" ADD CONSTRAINT "daily_followup_responses_assignment_id_daily_followup_assignments_id_fk" FOREIGN KEY ("assignment_id") REFERENCES "public"."daily_followup_assignments"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_followup_responses" ADD CONSTRAINT "daily_followup_responses_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_followup_templates" ADD CONSTRAINT "daily_followup_templates_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "deterioration_trends" ADD CONSTRAINT "deterioration_trends_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_health" ADD CONSTRAINT "device_health_device_connection_id_wearable_integrations_id_fk" FOREIGN KEY ("device_connection_id") REFERENCES "public"."wearable_integrations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_pairing_sessions" ADD CONSTRAINT "device_pairing_sessions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_pairing_sessions" ADD CONSTRAINT "device_pairing_sessions_device_model_id_device_models_id_fk" FOREIGN KEY ("device_model_id") REFERENCES "public"."device_models"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_pairing_sessions" ADD CONSTRAINT "device_pairing_sessions_result_device_connection_id_wearable_integrations_id_fk" FOREIGN KEY ("result_device_connection_id") REFERENCES "public"."wearable_integrations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_pairing_sessions" ADD CONSTRAINT "device_pairing_sessions_result_vendor_account_id_vendor_accounts_id_fk" FOREIGN KEY ("result_vendor_account_id") REFERENCES "public"."vendor_accounts"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_readings" ADD CONSTRAINT "device_readings_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_readings" ADD CONSTRAINT "device_readings_wearable_integration_id_wearable_integrations_id_fk" FOREIGN KEY ("wearable_integration_id") REFERENCES "public"."wearable_integrations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_sync_jobs" ADD CONSTRAINT "device_sync_jobs_vendor_account_id_vendor_accounts_id_fk" FOREIGN KEY ("vendor_account_id") REFERENCES "public"."vendor_accounts"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_sync_jobs" ADD CONSTRAINT "device_sync_jobs_device_connection_id_wearable_integrations_id_fk" FOREIGN KEY ("device_connection_id") REFERENCES "public"."wearable_integrations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "device_sync_jobs" ADD CONSTRAINT "device_sync_jobs_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "digital_biomarkers" ADD CONSTRAINT "digital_biomarkers_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_emails" ADD CONSTRAINT "doctor_emails_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_emails" ADD CONSTRAINT "doctor_emails_integration_id_doctor_integrations_id_fk" FOREIGN KEY ("integration_id") REFERENCES "public"."doctor_integrations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_emails" ADD CONSTRAINT "doctor_emails_linked_patient_id_users_id_fk" FOREIGN KEY ("linked_patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_integrations" ADD CONSTRAINT "doctor_integrations_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_assignments" ADD CONSTRAINT "doctor_patient_assignments_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_assignments" ADD CONSTRAINT "doctor_patient_assignments_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_assignments" ADD CONSTRAINT "doctor_patient_assignments_revoked_by_users_id_fk" FOREIGN KEY ("revoked_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_assignments" ADD CONSTRAINT "doctor_patient_assignments_assigned_by_users_id_fk" FOREIGN KEY ("assigned_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_consent_permissions" ADD CONSTRAINT "doctor_patient_consent_permissions_assignment_id_doctor_patient_assignments_id_fk" FOREIGN KEY ("assignment_id") REFERENCES "public"."doctor_patient_assignments"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_consent_permissions" ADD CONSTRAINT "doctor_patient_consent_permissions_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_consent_permissions" ADD CONSTRAINT "doctor_patient_consent_permissions_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_patient_consent_permissions" ADD CONSTRAINT "doctor_patient_consent_permissions_modified_by_users_id_fk" FOREIGN KEY ("modified_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_whatsapp_messages" ADD CONSTRAINT "doctor_whatsapp_messages_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_whatsapp_messages" ADD CONSTRAINT "doctor_whatsapp_messages_integration_id_doctor_integrations_id_fk" FOREIGN KEY ("integration_id") REFERENCES "public"."doctor_integrations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "doctor_whatsapp_messages" ADD CONSTRAINT "doctor_whatsapp_messages_linked_patient_id_users_id_fk" FOREIGN KEY ("linked_patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "dosage_change_requests" ADD CONSTRAINT "dosage_change_requests_medication_id_medications_id_fk" FOREIGN KEY ("medication_id") REFERENCES "public"."medications"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "dosage_change_requests" ADD CONSTRAINT "dosage_change_requests_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "dosage_change_requests" ADD CONSTRAINT "dosage_change_requests_reviewed_by_doctor_id_users_id_fk" FOREIGN KEY ("reviewed_by_doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "environmental_alerts" ADD CONSTRAINT "environmental_alerts_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "environmental_alerts" ADD CONSTRAINT "environmental_alerts_risk_score_id_patient_environment_risk_scores_id_fk" FOREIGN KEY ("risk_score_id") REFERENCES "public"."patient_environment_risk_scores"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "environmental_alerts" ADD CONSTRAINT "environmental_alerts_forecast_id_environmental_forecasts_id_fk" FOREIGN KEY ("forecast_id") REFERENCES "public"."environmental_forecasts"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "environmental_alerts" ADD CONSTRAINT "environmental_alerts_snapshot_id_environmental_data_snapshots_id_fk" FOREIGN KEY ("snapshot_id") REFERENCES "public"."environmental_data_snapshots"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "environmental_forecasts" ADD CONSTRAINT "environmental_forecasts_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "genetic_risk_flags" ADD CONSTRAINT "genetic_risk_flags_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "genetic_risk_flags" ADD CONSTRAINT "genetic_risk_flags_overridden_by_users_id_fk" FOREIGN KEY ("overridden_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "gmail_sync" ADD CONSTRAINT "gmail_sync_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "gmail_sync_logs" ADD CONSTRAINT "gmail_sync_logs_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "gmail_sync_logs" ADD CONSTRAINT "gmail_sync_logs_sync_id_gmail_sync_id_fk" FOREIGN KEY ("sync_id") REFERENCES "public"."gmail_sync"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "google_calendar_sync" ADD CONSTRAINT "google_calendar_sync_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "google_calendar_sync_logs" ADD CONSTRAINT "google_calendar_sync_logs_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "google_calendar_sync_logs" ADD CONSTRAINT "google_calendar_sync_logs_sync_id_google_calendar_sync_id_fk" FOREIGN KEY ("sync_id") REFERENCES "public"."google_calendar_sync"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_ai_recommendations" ADD CONSTRAINT "habit_ai_recommendations_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_ai_recommendations" ADD CONSTRAINT "habit_ai_recommendations_habit_id_habits_id_fk" FOREIGN KEY ("habit_id") REFERENCES "public"."habits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_ai_triggers" ADD CONSTRAINT "habit_ai_triggers_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_ai_triggers" ADD CONSTRAINT "habit_ai_triggers_habit_id_habits_id_fk" FOREIGN KEY ("habit_id") REFERENCES "public"."habits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_buddies" ADD CONSTRAINT "habit_buddies_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_buddies" ADD CONSTRAINT "habit_buddies_buddy_user_id_users_id_fk" FOREIGN KEY ("buddy_user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_buddies" ADD CONSTRAINT "habit_buddies_initiated_by_users_id_fk" FOREIGN KEY ("initiated_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_cbt_sessions" ADD CONSTRAINT "habit_cbt_sessions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_cbt_sessions" ADD CONSTRAINT "habit_cbt_sessions_related_habit_id_habits_id_fk" FOREIGN KEY ("related_habit_id") REFERENCES "public"."habits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_cbt_sessions" ADD CONSTRAINT "habit_cbt_sessions_related_quit_plan_id_habit_quit_plans_id_fk" FOREIGN KEY ("related_quit_plan_id") REFERENCES "public"."habit_quit_plans"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_coach_chats" ADD CONSTRAINT "habit_coach_chats_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_coach_chats" ADD CONSTRAINT "habit_coach_chats_related_habit_id_habits_id_fk" FOREIGN KEY ("related_habit_id") REFERENCES "public"."habits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_coach_chats" ADD CONSTRAINT "habit_coach_chats_related_quit_plan_id_habit_quit_plans_id_fk" FOREIGN KEY ("related_quit_plan_id") REFERENCES "public"."habit_quit_plans"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_cravings_log" ADD CONSTRAINT "habit_cravings_log_quit_plan_id_habit_quit_plans_id_fk" FOREIGN KEY ("quit_plan_id") REFERENCES "public"."habit_quit_plans"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_cravings_log" ADD CONSTRAINT "habit_cravings_log_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_encouragements" ADD CONSTRAINT "habit_encouragements_from_user_id_users_id_fk" FOREIGN KEY ("from_user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_encouragements" ADD CONSTRAINT "habit_encouragements_to_user_id_users_id_fk" FOREIGN KEY ("to_user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_encouragements" ADD CONSTRAINT "habit_encouragements_related_habit_id_habits_id_fk" FOREIGN KEY ("related_habit_id") REFERENCES "public"."habits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_journals" ADD CONSTRAINT "habit_journals_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_micro_steps" ADD CONSTRAINT "habit_micro_steps_habit_id_habits_id_fk" FOREIGN KEY ("habit_id") REFERENCES "public"."habits"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_mood_entries" ADD CONSTRAINT "habit_mood_entries_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_mood_entries" ADD CONSTRAINT "habit_mood_entries_associated_habit_id_habits_id_fk" FOREIGN KEY ("associated_habit_id") REFERENCES "public"."habits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_quit_plans" ADD CONSTRAINT "habit_quit_plans_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_relapse_log" ADD CONSTRAINT "habit_relapse_log_quit_plan_id_habit_quit_plans_id_fk" FOREIGN KEY ("quit_plan_id") REFERENCES "public"."habit_quit_plans"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_relapse_log" ADD CONSTRAINT "habit_relapse_log_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_reminders" ADD CONSTRAINT "habit_reminders_habit_id_habits_id_fk" FOREIGN KEY ("habit_id") REFERENCES "public"."habits"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_reminders" ADD CONSTRAINT "habit_reminders_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_rewards" ADD CONSTRAINT "habit_rewards_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_risk_alerts" ADD CONSTRAINT "habit_risk_alerts_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_risk_alerts" ADD CONSTRAINT "habit_risk_alerts_related_quit_plan_id_habit_quit_plans_id_fk" FOREIGN KEY ("related_quit_plan_id") REFERENCES "public"."habit_quit_plans"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_routines" ADD CONSTRAINT "habit_routines_habit_id_habits_id_fk" FOREIGN KEY ("habit_id") REFERENCES "public"."habits"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_routines" ADD CONSTRAINT "habit_routines_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "habit_routines" ADD CONSTRAINT "habit_routines_stacked_after_habits_id_fk" FOREIGN KEY ("stacked_after") REFERENCES "public"."habits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "health_section_analytics" ADD CONSTRAINT "health_section_analytics_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "infectious_events" ADD CONSTRAINT "infectious_events_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "infectious_events" ADD CONSTRAINT "infectious_events_related_condition_id_patient_conditions_id_fk" FOREIGN KEY ("related_condition_id") REFERENCES "public"."patient_conditions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "infectious_events" ADD CONSTRAINT "infectious_events_related_visit_id_patient_visits_id_fk" FOREIGN KEY ("related_visit_id") REFERENCES "public"."patient_visits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "infectious_events" ADD CONSTRAINT "infectious_events_location_id_healthcare_locations_id_fk" FOREIGN KEY ("location_id") REFERENCES "public"."healthcare_locations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "infectious_events" ADD CONSTRAINT "infectious_events_overridden_by_users_id_fk" FOREIGN KEY ("overridden_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_clinical_insights" ADD CONSTRAINT "lysa_clinical_insights_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_clinical_insights" ADD CONSTRAINT "lysa_clinical_insights_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_clinical_insights" ADD CONSTRAINT "lysa_clinical_insights_monitoring_assignment_id_lysa_monitoring_assignments_id_fk" FOREIGN KEY ("monitoring_assignment_id") REFERENCES "public"."lysa_monitoring_assignments"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_artifacts" ADD CONSTRAINT "lysa_monitoring_artifacts_monitoring_assignment_id_lysa_monitoring_assignments_id_fk" FOREIGN KEY ("monitoring_assignment_id") REFERENCES "public"."lysa_monitoring_assignments"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_artifacts" ADD CONSTRAINT "lysa_monitoring_artifacts_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_artifacts" ADD CONSTRAINT "lysa_monitoring_artifacts_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_artifacts" ADD CONSTRAINT "lysa_monitoring_artifacts_reviewed_by_users_id_fk" FOREIGN KEY ("reviewed_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_assignments" ADD CONSTRAINT "lysa_monitoring_assignments_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_assignments" ADD CONSTRAINT "lysa_monitoring_assignments_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_events" ADD CONSTRAINT "lysa_monitoring_events_monitoring_assignment_id_lysa_monitoring_assignments_id_fk" FOREIGN KEY ("monitoring_assignment_id") REFERENCES "public"."lysa_monitoring_assignments"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_events" ADD CONSTRAINT "lysa_monitoring_events_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_events" ADD CONSTRAINT "lysa_monitoring_events_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_events" ADD CONSTRAINT "lysa_monitoring_events_action_taken_by_users_id_fk" FOREIGN KEY ("action_taken_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "lysa_monitoring_events" ADD CONSTRAINT "lysa_monitoring_events_resolved_by_users_id_fk" FOREIGN KEY ("resolved_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_change_log" ADD CONSTRAINT "medication_change_log_medication_id_medications_id_fk" FOREIGN KEY ("medication_id") REFERENCES "public"."medications"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_change_log" ADD CONSTRAINT "medication_change_log_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_conflicts" ADD CONSTRAINT "medication_conflicts_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_conflicts" ADD CONSTRAINT "medication_conflicts_medication1_id_medications_id_fk" FOREIGN KEY ("medication1_id") REFERENCES "public"."medications"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_conflicts" ADD CONSTRAINT "medication_conflicts_medication2_id_medications_id_fk" FOREIGN KEY ("medication2_id") REFERENCES "public"."medications"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_conflicts" ADD CONSTRAINT "medication_conflicts_doctor1_id_users_id_fk" FOREIGN KEY ("doctor1_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_conflicts" ADD CONSTRAINT "medication_conflicts_doctor2_id_users_id_fk" FOREIGN KEY ("doctor2_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_drug_matches" ADD CONSTRAINT "medication_drug_matches_medication_id_medications_id_fk" FOREIGN KEY ("medication_id") REFERENCES "public"."medications"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medication_drug_matches" ADD CONSTRAINT "medication_drug_matches_drug_id_drugs_id_fk" FOREIGN KEY ("drug_id") REFERENCES "public"."drugs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mental_health_pattern_analysis" ADD CONSTRAINT "mental_health_pattern_analysis_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mental_health_pattern_analysis" ADD CONSTRAINT "mental_health_pattern_analysis_response_id_mental_health_responses_id_fk" FOREIGN KEY ("response_id") REFERENCES "public"."mental_health_responses"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mental_health_red_flags" ADD CONSTRAINT "mental_health_red_flags_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mental_health_red_flags" ADD CONSTRAINT "mental_health_red_flags_session_id_chat_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."chat_sessions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mental_health_red_flags" ADD CONSTRAINT "mental_health_red_flags_message_id_chat_messages_id_fk" FOREIGN KEY ("message_id") REFERENCES "public"."chat_messages"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mental_health_red_flags" ADD CONSTRAINT "mental_health_red_flags_reviewed_by_users_id_fk" FOREIGN KEY ("reviewed_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "mental_health_responses" ADD CONSTRAINT "mental_health_responses_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ml_training_consent" ADD CONSTRAINT "ml_training_consent_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ml_training_contributions" ADD CONSTRAINT "ml_training_contributions_consent_id_ml_training_consent_id_fk" FOREIGN KEY ("consent_id") REFERENCES "public"."ml_training_consent"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ml_training_contributions" ADD CONSTRAINT "ml_training_contributions_training_job_id_ml_training_jobs_id_fk" FOREIGN KEY ("training_job_id") REFERENCES "public"."ml_training_jobs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ml_training_contributions" ADD CONSTRAINT "ml_training_contributions_model_id_ml_models_id_fk" FOREIGN KEY ("model_id") REFERENCES "public"."ml_models"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "ml_training_jobs" ADD CONSTRAINT "ml_training_jobs_result_model_id_ml_models_id_fk" FOREIGN KEY ("result_model_id") REFERENCES "public"."ml_models"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "occupational_exposures" ADD CONSTRAINT "occupational_exposures_occupation_id_patient_occupations_id_fk" FOREIGN KEY ("occupation_id") REFERENCES "public"."patient_occupations"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "occupational_exposures" ADD CONSTRAINT "occupational_exposures_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "occupational_exposures" ADD CONSTRAINT "occupational_exposures_modified_by_users_id_fk" FOREIGN KEY ("modified_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "paintrack_sessions" ADD CONSTRAINT "paintrack_sessions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "passive_metrics" ADD CONSTRAINT "passive_metrics_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_conditions" ADD CONSTRAINT "patient_conditions_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_conditions" ADD CONSTRAINT "patient_conditions_diagnosed_by_users_id_fk" FOREIGN KEY ("diagnosed_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_consent_requests" ADD CONSTRAINT "patient_consent_requests_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_consent_requests" ADD CONSTRAINT "patient_consent_requests_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_environment_profiles" ADD CONSTRAINT "patient_environment_profiles_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_environment_risk_scores" ADD CONSTRAINT "patient_environment_risk_scores_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_environment_risk_scores" ADD CONSTRAINT "patient_environment_risk_scores_snapshot_id_environmental_data_snapshots_id_fk" FOREIGN KEY ("snapshot_id") REFERENCES "public"."environmental_data_snapshots"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_immunizations" ADD CONSTRAINT "patient_immunizations_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_immunizations" ADD CONSTRAINT "patient_immunizations_location_id_healthcare_locations_id_fk" FOREIGN KEY ("location_id") REFERENCES "public"."healthcare_locations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_immunizations" ADD CONSTRAINT "patient_immunizations_administered_by_users_id_fk" FOREIGN KEY ("administered_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_immunizations" ADD CONSTRAINT "patient_immunizations_overridden_by_users_id_fk" FOREIGN KEY ("overridden_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_occupations" ADD CONSTRAINT "patient_occupations_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_occupations" ADD CONSTRAINT "patient_occupations_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_occupations" ADD CONSTRAINT "patient_occupations_modified_by_users_id_fk" FOREIGN KEY ("modified_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_trigger_weights" ADD CONSTRAINT "patient_trigger_weights_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_visits" ADD CONSTRAINT "patient_visits_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "patient_visits" ADD CONSTRAINT "patient_visits_location_id_healthcare_locations_id_fk" FOREIGN KEY ("location_id") REFERENCES "public"."healthcare_locations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "prescriptions" ADD CONSTRAINT "prescriptions_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "prescriptions" ADD CONSTRAINT "prescriptions_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "prescriptions" ADD CONSTRAINT "prescriptions_drug_id_drugs_id_fk" FOREIGN KEY ("drug_id") REFERENCES "public"."drugs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "prescriptions" ADD CONSTRAINT "prescriptions_medication_id_medications_id_fk" FOREIGN KEY ("medication_id") REFERENCES "public"."medications"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_alerts" ADD CONSTRAINT "research_alerts_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_alerts" ADD CONSTRAINT "research_alerts_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_alerts" ADD CONSTRAINT "research_alerts_acknowledged_by_users_id_fk" FOREIGN KEY ("acknowledged_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_alerts" ADD CONSTRAINT "research_alerts_resolved_by_users_id_fk" FOREIGN KEY ("resolved_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_analysis_reports" ADD CONSTRAINT "research_analysis_reports_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_analysis_reports" ADD CONSTRAINT "research_analysis_reports_cohort_id_research_cohorts_id_fk" FOREIGN KEY ("cohort_id") REFERENCES "public"."research_cohorts"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_analysis_reports" ADD CONSTRAINT "research_analysis_reports_snapshot_id_research_data_snapshots_id_fk" FOREIGN KEY ("snapshot_id") REFERENCES "public"."research_data_snapshots"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_analysis_reports" ADD CONSTRAINT "research_analysis_reports_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_audit_logs" ADD CONSTRAINT "research_audit_logs_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_cohorts" ADD CONSTRAINT "research_cohorts_project_id_research_projects_id_fk" FOREIGN KEY ("project_id") REFERENCES "public"."research_projects"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_cohorts" ADD CONSTRAINT "research_cohorts_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_data_consent" ADD CONSTRAINT "research_data_consent_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_data_snapshots" ADD CONSTRAINT "research_data_snapshots_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_data_snapshots" ADD CONSTRAINT "research_data_snapshots_cohort_id_research_cohorts_id_fk" FOREIGN KEY ("cohort_id") REFERENCES "public"."research_cohorts"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_data_snapshots" ADD CONSTRAINT "research_data_snapshots_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_environmental_exposures" ADD CONSTRAINT "research_environmental_exposures_location_id_research_locations_id_fk" FOREIGN KEY ("location_id") REFERENCES "public"."research_locations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_immune_markers" ADD CONSTRAINT "research_immune_markers_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_immune_markers" ADD CONSTRAINT "research_immune_markers_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_immune_markers" ADD CONSTRAINT "research_immune_markers_visit_id_research_visits_id_fk" FOREIGN KEY ("visit_id") REFERENCES "public"."research_visits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_measurements" ADD CONSTRAINT "research_measurements_visit_id_research_visits_id_fk" FOREIGN KEY ("visit_id") REFERENCES "public"."research_visits"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_measurements" ADD CONSTRAINT "research_measurements_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_measurements" ADD CONSTRAINT "research_measurements_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_patient_locations" ADD CONSTRAINT "research_patient_locations_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_patient_locations" ADD CONSTRAINT "research_patient_locations_location_id_research_locations_id_fk" FOREIGN KEY ("location_id") REFERENCES "public"."research_locations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_projects" ADD CONSTRAINT "research_projects_owner_id_users_id_fk" FOREIGN KEY ("owner_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_studies" ADD CONSTRAINT "research_studies_project_id_research_projects_id_fk" FOREIGN KEY ("project_id") REFERENCES "public"."research_projects"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_studies" ADD CONSTRAINT "research_studies_owner_user_id_users_id_fk" FOREIGN KEY ("owner_user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_visits" ADD CONSTRAINT "research_visits_enrollment_id_study_enrollments_id_fk" FOREIGN KEY ("enrollment_id") REFERENCES "public"."study_enrollments"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_visits" ADD CONSTRAINT "research_visits_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "research_visits" ADD CONSTRAINT "research_visits_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sentiment_analysis" ADD CONSTRAINT "sentiment_analysis_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "session_metrics" ADD CONSTRAINT "session_metrics_session_id_paintrack_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."paintrack_sessions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "study_enrollments" ADD CONSTRAINT "study_enrollments_study_id_research_studies_id_fk" FOREIGN KEY ("study_id") REFERENCES "public"."research_studies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "study_enrollments" ADD CONSTRAINT "study_enrollments_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "symptom_checkins" ADD CONSTRAINT "symptom_checkins_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "symptom_checkins" ADD CONSTRAINT "symptom_checkins_session_id_chat_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."chat_sessions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "symptom_environment_correlations" ADD CONSTRAINT "symptom_environment_correlations_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tool_executions" ADD CONSTRAINT "tool_executions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tool_executions" ADD CONSTRAINT "tool_executions_conversation_id_agent_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."agent_conversations"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tool_executions" ADD CONSTRAINT "tool_executions_message_id_agent_messages_id_fk" FOREIGN KEY ("message_id") REFERENCES "public"."agent_messages"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tool_executions" ADD CONSTRAINT "tool_executions_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tool_executions" ADD CONSTRAINT "tool_executions_doctor_id_users_id_fk" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "trend_reports" ADD CONSTRAINT "trend_reports_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "trend_reports" ADD CONSTRAINT "trend_reports_reviewed_by_users_id_fk" FOREIGN KEY ("reviewed_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_presence" ADD CONSTRAINT "user_presence_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "vendor_accounts" ADD CONSTRAINT "vendor_accounts_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "video_exam_segments" ADD CONSTRAINT "video_exam_segments_session_id_video_exam_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."video_exam_sessions"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "video_exam_segments" ADD CONSTRAINT "video_exam_segments_uploaded_by_users_id_fk" FOREIGN KEY ("uploaded_by") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "video_exam_sessions" ADD CONSTRAINT "video_exam_sessions_patient_id_users_id_fk" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "audit_actor_idx" ON "agent_audit_logs" USING btree ("actor_type","actor_id");--> statement-breakpoint
CREATE INDEX "audit_action_idx" ON "agent_audit_logs" USING btree ("action");--> statement-breakpoint
CREATE INDEX "audit_object_idx" ON "agent_audit_logs" USING btree ("object_type","object_id");--> statement-breakpoint
CREATE INDEX "audit_patient_idx" ON "agent_audit_logs" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "audit_timestamp_idx" ON "agent_audit_logs" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "audit_phi_idx" ON "agent_audit_logs" USING btree ("phi_accessed");--> statement-breakpoint
CREATE INDEX "conv_type_idx" ON "agent_conversations" USING btree ("conversation_type");--> statement-breakpoint
CREATE INDEX "conv_participant1_idx" ON "agent_conversations" USING btree ("participant1_type","participant1_id");--> statement-breakpoint
CREATE INDEX "conv_participant2_idx" ON "agent_conversations" USING btree ("participant2_type","participant2_id");--> statement-breakpoint
CREATE INDEX "conv_patient_idx" ON "agent_conversations" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "conv_doctor_idx" ON "agent_conversations" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "conv_assignment_idx" ON "agent_conversations" USING btree ("assignment_id");--> statement-breakpoint
CREATE INDEX "conv_last_message_idx" ON "agent_conversations" USING btree ("last_message_at");--> statement-breakpoint
CREATE INDEX "memory_agent_idx" ON "agent_memory" USING btree ("agent_id");--> statement-breakpoint
CREATE INDEX "memory_patient_idx" ON "agent_memory" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "memory_type_idx" ON "agent_memory" USING btree ("memory_type");--> statement-breakpoint
CREATE INDEX "memory_expires_idx" ON "agent_memory" USING btree ("expires_at");--> statement-breakpoint
CREATE INDEX "memory_importance_idx" ON "agent_memory" USING btree ("importance");--> statement-breakpoint
CREATE INDEX "msg_conversation_idx" ON "agent_messages" USING btree ("conversation_id");--> statement-breakpoint
CREATE INDEX "msg_from_idx" ON "agent_messages" USING btree ("from_type","from_id");--> statement-breakpoint
CREATE INDEX "msg_sender_role_idx" ON "agent_messages" USING btree ("sender_role");--> statement-breakpoint
CREATE INDEX "msg_id_idx" ON "agent_messages" USING btree ("msg_id");--> statement-breakpoint
CREATE INDEX "msg_created_idx" ON "agent_messages" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "msg_tool_call_idx" ON "agent_messages" USING btree ("tool_call_id");--> statement-breakpoint
CREATE INDEX "msg_approval_idx" ON "agent_messages" USING btree ("requires_approval","approval_status");--> statement-breakpoint
CREATE INDEX "task_agent_idx" ON "agent_tasks" USING btree ("agent_id");--> statement-breakpoint
CREATE INDEX "task_user_idx" ON "agent_tasks" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "task_status_idx" ON "agent_tasks" USING btree ("status");--> statement-breakpoint
CREATE INDEX "task_scheduled_idx" ON "agent_tasks" USING btree ("scheduled_at");--> statement-breakpoint
CREATE INDEX "task_type_idx" ON "agent_tasks" USING btree ("task_type");--> statement-breakpoint
CREATE INDEX "agent_tool_idx" ON "agent_tool_assignments" USING btree ("agent_id","tool_id");--> statement-breakpoint
CREATE INDEX "ai_engagement_metrics_patient_period_idx" ON "ai_engagement_metrics" USING btree ("patient_id","period_start");--> statement-breakpoint
CREATE INDEX "ai_health_alerts_patient_status_idx" ON "ai_health_alerts" USING btree ("patient_id","status");--> statement-breakpoint
CREATE INDEX "ai_health_alerts_severity_idx" ON "ai_health_alerts" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "ai_health_alerts_type_idx" ON "ai_health_alerts" USING btree ("alert_type");--> statement-breakpoint
CREATE INDEX "ai_health_alerts_created_at_idx" ON "ai_health_alerts" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "ai_qol_metrics_patient_date_idx" ON "ai_qol_metrics" USING btree ("patient_id","recorded_at");--> statement-breakpoint
CREATE INDEX "ai_qol_metrics_wellness_idx" ON "ai_qol_metrics" USING btree ("wellness_index");--> statement-breakpoint
CREATE INDEX "ai_trend_metrics_patient_metric_idx" ON "ai_trend_metrics" USING btree ("patient_id","metric_name");--> statement-breakpoint
CREATE INDEX "ai_trend_metrics_recorded_at_idx" ON "ai_trend_metrics" USING btree ("recorded_at");--> statement-breakpoint
CREATE INDEX "ai_trend_metrics_zscore_idx" ON "ai_trend_metrics" USING btree ("z_score");--> statement-breakpoint
CREATE INDEX "analysis_job_status_idx" ON "analysis_jobs" USING btree ("status");--> statement-breakpoint
CREATE INDEX "analysis_job_study_idx" ON "analysis_jobs" USING btree ("study_id");--> statement-breakpoint
CREATE INDEX "analysis_job_type_idx" ON "analysis_jobs" USING btree ("job_type");--> statement-breakpoint
CREATE INDEX "approval_requester_idx" ON "approval_queue" USING btree ("requester_id","requester_type");--> statement-breakpoint
CREATE INDEX "approval_approver_idx" ON "approval_queue" USING btree ("approver_id");--> statement-breakpoint
CREATE INDEX "approval_role_idx" ON "approval_queue" USING btree ("approver_role");--> statement-breakpoint
CREATE INDEX "approval_patient_idx" ON "approval_queue" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "approval_status_idx" ON "approval_queue" USING btree ("status");--> statement-breakpoint
CREATE INDEX "approval_urgency_idx" ON "approval_queue" USING btree ("urgency");--> statement-breakpoint
CREATE INDEX "approval_expires_idx" ON "approval_queue" USING btree ("expires_at");--> statement-breakpoint
CREATE INDEX "approval_created_idx" ON "approval_queue" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "behavior_alerts_patient_severity_idx" ON "behavior_alerts" USING btree ("patient_id","severity");--> statement-breakpoint
CREATE INDEX "behavior_alerts_type_idx" ON "behavior_alerts" USING btree ("alert_type");--> statement-breakpoint
CREATE INDEX "behavior_alerts_acknowledged_idx" ON "behavior_alerts" USING btree ("acknowledged");--> statement-breakpoint
CREATE INDEX "behavior_checkins_patient_idx" ON "behavior_checkins" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "behavior_checkins_completed_idx" ON "behavior_checkins" USING btree ("completed_at");--> statement-breakpoint
CREATE INDEX "behavior_checkins_skipped_idx" ON "behavior_checkins" USING btree ("skipped");--> statement-breakpoint
CREATE INDEX "behavior_metrics_patient_date_idx" ON "behavior_metrics" USING btree ("patient_id","date");--> statement-breakpoint
CREATE INDEX "behavior_risk_scores_patient_calc_idx" ON "behavior_risk_scores" USING btree ("patient_id","calculated_at");--> statement-breakpoint
CREATE INDEX "behavior_risk_scores_risk_level_idx" ON "behavior_risk_scores" USING btree ("risk_level");--> statement-breakpoint
CREATE INDEX "clinician_notes_session_idx" ON "clinician_notes" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "clinician_notes_clinician_idx" ON "clinician_notes" USING btree ("clinician_id");--> statement-breakpoint
CREATE INDEX "clinician_workload_clinician_period_idx" ON "clinician_workload_metrics" USING btree ("clinician_id","period_start");--> statement-breakpoint
CREATE INDEX "cognitive_tests_patient_type_idx" ON "cognitive_tests" USING btree ("patient_id","test_type");--> statement-breakpoint
CREATE INDEX "cognitive_tests_anomaly_idx" ON "cognitive_tests" USING btree ("anomaly_detected");--> statement-breakpoint
CREATE INDEX "trigger_condition_idx" ON "condition_trigger_mappings" USING btree ("condition_code");--> statement-breakpoint
CREATE INDEX "trigger_factor_idx" ON "condition_trigger_mappings" USING btree ("factor_type");--> statement-breakpoint
CREATE INDEX "followup_assignment_template_idx" ON "daily_followup_assignments" USING btree ("template_id");--> statement-breakpoint
CREATE INDEX "followup_assignment_patient_idx" ON "daily_followup_assignments" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "followup_assignment_study_idx" ON "daily_followup_assignments" USING btree ("study_id");--> statement-breakpoint
CREATE INDEX "followup_assignment_active_idx" ON "daily_followup_assignments" USING btree ("is_active");--> statement-breakpoint
CREATE INDEX "followup_response_assignment_idx" ON "daily_followup_responses" USING btree ("assignment_id");--> statement-breakpoint
CREATE INDEX "followup_response_patient_idx" ON "daily_followup_responses" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "followup_response_date_idx" ON "daily_followup_responses" USING btree ("response_date");--> statement-breakpoint
CREATE INDEX "followup_template_category_idx" ON "daily_followup_templates" USING btree ("category");--> statement-breakpoint
CREATE INDEX "followup_template_active_idx" ON "daily_followup_templates" USING btree ("is_active");--> statement-breakpoint
CREATE INDEX "data_snapshot_created_by_idx" ON "data_snapshots" USING btree ("created_by");--> statement-breakpoint
CREATE INDEX "deterioration_trends_patient_type_idx" ON "deterioration_trends" USING btree ("patient_id","trend_type");--> statement-breakpoint
CREATE INDEX "deterioration_trends_severity_idx" ON "deterioration_trends" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "deterioration_trends_alert_idx" ON "deterioration_trends" USING btree ("alert_generated");--> statement-breakpoint
CREATE INDEX "device_audit_actor_idx" ON "device_data_audit_log" USING btree ("actor_id");--> statement-breakpoint
CREATE INDEX "device_audit_action_idx" ON "device_data_audit_log" USING btree ("action");--> statement-breakpoint
CREATE INDEX "device_audit_resource_idx" ON "device_data_audit_log" USING btree ("resource_type","resource_id");--> statement-breakpoint
CREATE INDEX "device_audit_patient_idx" ON "device_data_audit_log" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "device_audit_created_idx" ON "device_data_audit_log" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "device_health_device_idx" ON "device_health" USING btree ("device_connection_id");--> statement-breakpoint
CREATE INDEX "device_health_status_idx" ON "device_health" USING btree ("status");--> statement-breakpoint
CREATE INDEX "device_health_last_seen_idx" ON "device_health" USING btree ("last_seen_at");--> statement-breakpoint
CREATE INDEX "device_models_vendor_idx" ON "device_models" USING btree ("vendor_id");--> statement-breakpoint
CREATE INDEX "device_models_type_idx" ON "device_models" USING btree ("device_type");--> statement-breakpoint
CREATE INDEX "device_models_active_idx" ON "device_models" USING btree ("is_active");--> statement-breakpoint
CREATE INDEX "device_pairing_user_idx" ON "device_pairing_sessions" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "device_pairing_status_idx" ON "device_pairing_sessions" USING btree ("session_status");--> statement-breakpoint
CREATE INDEX "device_pairing_oauth_state_idx" ON "device_pairing_sessions" USING btree ("oauth_state");--> statement-breakpoint
CREATE INDEX "device_pairing_expiry_idx" ON "device_pairing_sessions" USING btree ("expires_at");--> statement-breakpoint
CREATE INDEX "device_readings_patient_date_idx" ON "device_readings" USING btree ("patient_id","recorded_at");--> statement-breakpoint
CREATE INDEX "device_readings_device_type_idx" ON "device_readings" USING btree ("device_type");--> statement-breakpoint
CREATE INDEX "device_readings_source_idx" ON "device_readings" USING btree ("source");--> statement-breakpoint
CREATE INDEX "device_sync_jobs_status_idx" ON "device_sync_jobs" USING btree ("status");--> statement-breakpoint
CREATE INDEX "device_sync_jobs_scheduled_idx" ON "device_sync_jobs" USING btree ("scheduled_for");--> statement-breakpoint
CREATE INDEX "device_sync_jobs_vendor_idx" ON "device_sync_jobs" USING btree ("vendor_account_id");--> statement-breakpoint
CREATE INDEX "digital_biomarkers_patient_date_idx" ON "digital_biomarkers" USING btree ("patient_id","date");--> statement-breakpoint
CREATE INDEX "digital_biomarkers_mobility_drop_idx" ON "digital_biomarkers" USING btree ("mobility_drop_detected");--> statement-breakpoint
CREATE INDEX "doctor_emails_doctor_idx" ON "doctor_emails" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "doctor_emails_thread_idx" ON "doctor_emails" USING btree ("thread_id");--> statement-breakpoint
CREATE INDEX "doctor_emails_category_idx" ON "doctor_emails" USING btree ("ai_category");--> statement-breakpoint
CREATE INDEX "doctor_emails_patient_idx" ON "doctor_emails" USING btree ("linked_patient_id");--> statement-breakpoint
CREATE INDEX "doctor_emails_received_idx" ON "doctor_emails" USING btree ("received_at");--> statement-breakpoint
CREATE INDEX "doctor_integrations_doctor_idx" ON "doctor_integrations" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "doctor_integrations_type_idx" ON "doctor_integrations" USING btree ("integration_type");--> statement-breakpoint
CREATE INDEX "doctor_integrations_status_idx" ON "doctor_integrations" USING btree ("status");--> statement-breakpoint
CREATE INDEX "doctor_integrations_unique_idx" ON "doctor_integrations" USING btree ("doctor_id","integration_type");--> statement-breakpoint
CREATE INDEX "doctor_patient_idx" ON "doctor_patient_assignments" USING btree ("doctor_id","patient_id");--> statement-breakpoint
CREATE INDEX "doctor_active_assignments_idx" ON "doctor_patient_assignments" USING btree ("doctor_id","status");--> statement-breakpoint
CREATE INDEX "patient_active_assignments_idx" ON "doctor_patient_assignments" USING btree ("patient_id","status");--> statement-breakpoint
CREATE INDEX "unique_active_doctor_patient" ON "doctor_patient_assignments" USING btree ("doctor_id","patient_id","status");--> statement-breakpoint
CREATE INDEX "consent_permissions_assignment_idx" ON "doctor_patient_consent_permissions" USING btree ("assignment_id");--> statement-breakpoint
CREATE INDEX "consent_permissions_doctor_patient_idx" ON "doctor_patient_consent_permissions" USING btree ("doctor_id","patient_id");--> statement-breakpoint
CREATE INDEX "doctor_wa_doctor_idx" ON "doctor_whatsapp_messages" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "doctor_wa_conversation_idx" ON "doctor_whatsapp_messages" USING btree ("wa_conversation_id");--> statement-breakpoint
CREATE INDEX "doctor_wa_patient_idx" ON "doctor_whatsapp_messages" USING btree ("linked_patient_id");--> statement-breakpoint
CREATE INDEX "doctor_wa_received_idx" ON "doctor_whatsapp_messages" USING btree ("received_at");--> statement-breakpoint
CREATE INDEX "embedding_entity_type_idx" ON "entity_embeddings" USING btree ("entity_type");--> statement-breakpoint
CREATE INDEX "embedding_method_idx" ON "entity_embeddings" USING btree ("method");--> statement-breakpoint
CREATE INDEX "env_alert_patient_status_idx" ON "environmental_alerts" USING btree ("patient_id","status");--> statement-breakpoint
CREATE INDEX "env_alert_severity_idx" ON "environmental_alerts" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "env_alert_created_idx" ON "environmental_alerts" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "env_snapshot_zip_time_idx" ON "environmental_data_snapshots" USING btree ("zip_code","measured_at");--> statement-breakpoint
CREATE INDEX "env_snapshot_measured_idx" ON "environmental_data_snapshots" USING btree ("measured_at");--> statement-breakpoint
CREATE INDEX "forecast_patient_time_idx" ON "environmental_forecasts" USING btree ("patient_id","generated_at");--> statement-breakpoint
CREATE INDEX "forecast_horizon_idx" ON "environmental_forecasts" USING btree ("forecast_horizon");--> statement-breakpoint
CREATE INDEX "pipeline_job_type_status_idx" ON "environmental_pipeline_jobs" USING btree ("job_type","status");--> statement-breakpoint
CREATE INDEX "pipeline_job_created_idx" ON "environmental_pipeline_jobs" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "genetic_flag_patient_idx" ON "genetic_risk_flags" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "genetic_flag_name_idx" ON "genetic_risk_flags" USING btree ("flag_name");--> statement-breakpoint
CREATE INDEX "genetic_flag_type_idx" ON "genetic_risk_flags" USING btree ("flag_type");--> statement-breakpoint
CREATE INDEX "gmail_sync_doctor_idx" ON "gmail_sync" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "gmail_sync_logs_doctor_idx" ON "gmail_sync_logs" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "gmail_sync_logs_created_idx" ON "gmail_sync_logs" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "gmail_sync_logs_action_idx" ON "gmail_sync_logs" USING btree ("action");--> statement-breakpoint
CREATE INDEX "google_calendar_sync_doctor_idx" ON "google_calendar_sync" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "google_calendar_sync_logs_doctor_idx" ON "google_calendar_sync_logs" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "google_calendar_sync_logs_created_idx" ON "google_calendar_sync_logs" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "health_analytics_patient_section_idx" ON "health_section_analytics" USING btree ("patient_id","health_section");--> statement-breakpoint
CREATE INDEX "health_analytics_date_idx" ON "health_section_analytics" USING btree ("analysis_date");--> statement-breakpoint
CREATE INDEX "health_analytics_risk_idx" ON "health_section_analytics" USING btree ("risk_level");--> statement-breakpoint
CREATE INDEX "infectious_event_patient_idx" ON "infectious_events" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "infectious_event_type_idx" ON "infectious_events" USING btree ("infection_type");--> statement-breakpoint
CREATE INDEX "infectious_event_onset_idx" ON "infectious_events" USING btree ("onset_date");--> statement-breakpoint
CREATE INDEX "infectious_event_severity_idx" ON "infectious_events" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "lysa_insights_patient_idx" ON "lysa_clinical_insights" USING btree ("patient_id","insight_type");--> statement-breakpoint
CREATE INDEX "lysa_insights_doctor_idx" ON "lysa_clinical_insights" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "lysa_insights_severity_idx" ON "lysa_clinical_insights" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "lysa_insights_status_idx" ON "lysa_clinical_insights" USING btree ("status");--> statement-breakpoint
CREATE INDEX "lysa_insights_created_at_idx" ON "lysa_clinical_insights" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "lysa_artifacts_assignment_idx" ON "lysa_monitoring_artifacts" USING btree ("monitoring_assignment_id");--> statement-breakpoint
CREATE INDEX "lysa_artifacts_patient_idx" ON "lysa_monitoring_artifacts" USING btree ("patient_id","artifact_type");--> statement-breakpoint
CREATE INDEX "lysa_artifacts_doctor_idx" ON "lysa_monitoring_artifacts" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "lysa_artifacts_created_at_idx" ON "lysa_monitoring_artifacts" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "lysa_artifacts_period_idx" ON "lysa_monitoring_artifacts" USING btree ("period_start","period_end");--> statement-breakpoint
CREATE INDEX "lysa_monitoring_doctor_patient_idx" ON "lysa_monitoring_assignments" USING btree ("doctor_id","patient_id");--> statement-breakpoint
CREATE INDEX "lysa_monitoring_doctor_active_idx" ON "lysa_monitoring_assignments" USING btree ("doctor_id","is_active");--> statement-breakpoint
CREATE INDEX "lysa_monitoring_patient_active_idx" ON "lysa_monitoring_assignments" USING btree ("patient_id","is_active");--> statement-breakpoint
CREATE INDEX "lysa_monitoring_next_check_idx" ON "lysa_monitoring_assignments" USING btree ("next_scheduled_check");--> statement-breakpoint
CREATE INDEX "lysa_events_assignment_idx" ON "lysa_monitoring_events" USING btree ("monitoring_assignment_id");--> statement-breakpoint
CREATE INDEX "lysa_events_patient_idx" ON "lysa_monitoring_events" USING btree ("patient_id","event_type");--> statement-breakpoint
CREATE INDEX "lysa_events_doctor_idx" ON "lysa_monitoring_events" USING btree ("doctor_id");--> statement-breakpoint
CREATE INDEX "lysa_events_severity_idx" ON "lysa_monitoring_events" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "lysa_events_status_idx" ON "lysa_monitoring_events" USING btree ("status");--> statement-breakpoint
CREATE INDEX "lysa_events_created_at_idx" ON "lysa_monitoring_events" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "mh_analysis_patient_idx" ON "mental_health_pattern_analysis" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "mh_analysis_response_idx" ON "mental_health_pattern_analysis" USING btree ("response_id");--> statement-breakpoint
CREATE INDEX "mh_analysis_type_idx" ON "mental_health_pattern_analysis" USING btree ("analysis_type");--> statement-breakpoint
CREATE INDEX "mental_health_red_flags_user_id_idx" ON "mental_health_red_flags" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "mental_health_red_flags_session_id_idx" ON "mental_health_red_flags" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "mental_health_red_flags_created_at_idx" ON "mental_health_red_flags" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "mh_responses_patient_type_idx" ON "mental_health_responses" USING btree ("patient_id","questionnaire_type");--> statement-breakpoint
CREATE INDEX "mh_responses_severity_idx" ON "mental_health_responses" USING btree ("severity_level");--> statement-breakpoint
CREATE INDEX "mh_responses_crisis_idx" ON "mental_health_responses" USING btree ("crisis_detected");--> statement-breakpoint
CREATE INDEX "mh_responses_completed_idx" ON "mental_health_responses" USING btree ("completed_at");--> statement-breakpoint
CREATE INDEX "ml_extraction_requester_idx" ON "ml_extraction_audit_log" USING btree ("requester_id");--> statement-breakpoint
CREATE INDEX "ml_extraction_policy_idx" ON "ml_extraction_audit_log" USING btree ("policy_id");--> statement-breakpoint
CREATE INDEX "ml_models_name_idx" ON "ml_models" USING btree ("model_name");--> statement-breakpoint
CREATE INDEX "ml_models_version_idx" ON "ml_models" USING btree ("model_name","version");--> statement-breakpoint
CREATE INDEX "ml_models_status_idx" ON "ml_models" USING btree ("status");--> statement-breakpoint
CREATE INDEX "ml_models_active_idx" ON "ml_models" USING btree ("is_active");--> statement-breakpoint
CREATE INDEX "ml_audit_event_idx" ON "ml_training_audit_log" USING btree ("event_type");--> statement-breakpoint
CREATE INDEX "ml_audit_actor_idx" ON "ml_training_audit_log" USING btree ("actor_id");--> statement-breakpoint
CREATE INDEX "ml_audit_resource_idx" ON "ml_training_audit_log" USING btree ("resource_type","resource_id");--> statement-breakpoint
CREATE INDEX "ml_audit_patient_idx" ON "ml_training_audit_log" USING btree ("patient_id_hash");--> statement-breakpoint
CREATE INDEX "ml_audit_created_idx" ON "ml_training_audit_log" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "ml_consent_patient_idx" ON "ml_training_consent" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "ml_consent_enabled_idx" ON "ml_training_consent" USING btree ("consent_enabled");--> statement-breakpoint
CREATE INDEX "ml_contrib_patient_idx" ON "ml_training_contributions" USING btree ("patient_id_hash");--> statement-breakpoint
CREATE INDEX "ml_contrib_job_idx" ON "ml_training_contributions" USING btree ("training_job_id");--> statement-breakpoint
CREATE INDEX "ml_contrib_model_idx" ON "ml_training_contributions" USING btree ("model_id");--> statement-breakpoint
CREATE INDEX "ml_jobs_status_idx" ON "ml_training_jobs" USING btree ("status");--> statement-breakpoint
CREATE INDEX "ml_jobs_model_idx" ON "ml_training_jobs" USING btree ("model_name");--> statement-breakpoint
CREATE INDEX "ml_jobs_priority_idx" ON "ml_training_jobs" USING btree ("priority","queued_at");--> statement-breakpoint
CREATE INDEX "exposure_occupation_idx" ON "occupational_exposures" USING btree ("occupation_id");--> statement-breakpoint
CREATE INDEX "exposure_patient_idx" ON "occupational_exposures" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "exposure_type_idx" ON "occupational_exposures" USING btree ("exposure_type");--> statement-breakpoint
CREATE INDEX "paintrack_user_idx" ON "paintrack_sessions" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "paintrack_module_idx" ON "paintrack_sessions" USING btree ("module");--> statement-breakpoint
CREATE INDEX "paintrack_joint_idx" ON "paintrack_sessions" USING btree ("joint");--> statement-breakpoint
CREATE INDEX "paintrack_status_idx" ON "paintrack_sessions" USING btree ("status");--> statement-breakpoint
CREATE INDEX "passive_metrics_user_date_idx" ON "passive_metrics" USING btree ("user_id","date");--> statement-breakpoint
CREATE INDEX "patient_condition_patient_idx" ON "patient_conditions" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "patient_condition_code_idx" ON "patient_conditions" USING btree ("condition_code");--> statement-breakpoint
CREATE INDEX "patient_condition_status_idx" ON "patient_conditions" USING btree ("status");--> statement-breakpoint
CREATE INDEX "patient_condition_category_idx" ON "patient_conditions" USING btree ("condition_category");--> statement-breakpoint
CREATE INDEX "doctor_pending_requests_idx" ON "patient_consent_requests" USING btree ("doctor_id","status");--> statement-breakpoint
CREATE INDEX "patient_pending_requests_idx" ON "patient_consent_requests" USING btree ("patient_id","status");--> statement-breakpoint
CREATE INDEX "env_profile_patient_idx" ON "patient_environment_profiles" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "env_profile_zip_idx" ON "patient_environment_profiles" USING btree ("zip_code");--> statement-breakpoint
CREATE INDEX "risk_score_patient_time_idx" ON "patient_environment_risk_scores" USING btree ("patient_id","computed_at");--> statement-breakpoint
CREATE INDEX "risk_score_level_idx" ON "patient_environment_risk_scores" USING btree ("risk_level");--> statement-breakpoint
CREATE INDEX "immunization_patient_idx" ON "patient_immunizations" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "immunization_vaccine_idx" ON "patient_immunizations" USING btree ("vaccine_code");--> statement-breakpoint
CREATE INDEX "immunization_date_idx" ON "patient_immunizations" USING btree ("administration_date");--> statement-breakpoint
CREATE INDEX "occupation_patient_idx" ON "patient_occupations" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "occupation_industry_idx" ON "patient_occupations" USING btree ("industry");--> statement-breakpoint
CREATE INDEX "occupation_current_idx" ON "patient_occupations" USING btree ("is_current");--> statement-breakpoint
CREATE INDEX "patient_trigger_idx" ON "patient_trigger_weights" USING btree ("patient_id","factor_type");--> statement-breakpoint
CREATE INDEX "patient_visit_patient_idx" ON "patient_visits" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "patient_visit_date_idx" ON "patient_visits" USING btree ("admission_date");--> statement-breakpoint
CREATE INDEX "patient_visit_type_idx" ON "patient_visits" USING btree ("visit_type");--> statement-breakpoint
CREATE INDEX "protocol_version_audit_protocol_idx" ON "protocol_version_audit_log" USING btree ("protocol_id");--> statement-breakpoint
CREATE INDEX "protocol_version_audit_action_idx" ON "protocol_version_audit_log" USING btree ("action");--> statement-breakpoint
CREATE INDEX "public_dataset_source_idx" ON "public_dataset_registry" USING btree ("source");--> statement-breakpoint
CREATE INDEX "public_dataset_status_idx" ON "public_dataset_registry" USING btree ("download_status");--> statement-breakpoint
CREATE INDEX "research_alert_patient_idx" ON "research_alerts" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "research_alert_study_idx" ON "research_alerts" USING btree ("study_id");--> statement-breakpoint
CREATE INDEX "research_alert_type_idx" ON "research_alerts" USING btree ("alert_type");--> statement-breakpoint
CREATE INDEX "research_alert_status_idx" ON "research_alerts" USING btree ("status");--> statement-breakpoint
CREATE INDEX "research_alert_severity_idx" ON "research_alerts" USING btree ("severity");--> statement-breakpoint
CREATE INDEX "analysis_report_study_idx" ON "research_analysis_reports" USING btree ("study_id");--> statement-breakpoint
CREATE INDEX "analysis_report_cohort_idx" ON "research_analysis_reports" USING btree ("cohort_id");--> statement-breakpoint
CREATE INDEX "analysis_report_type_idx" ON "research_analysis_reports" USING btree ("analysis_type");--> statement-breakpoint
CREATE INDEX "analysis_report_status_idx" ON "research_analysis_reports" USING btree ("status");--> statement-breakpoint
CREATE INDEX "research_audit_user_idx" ON "research_audit_logs" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "research_audit_action_idx" ON "research_audit_logs" USING btree ("action_type");--> statement-breakpoint
CREATE INDEX "research_audit_object_idx" ON "research_audit_logs" USING btree ("object_type","object_id");--> statement-breakpoint
CREATE INDEX "research_audit_created_idx" ON "research_audit_logs" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "research_cohort_project_idx" ON "research_cohorts" USING btree ("project_id");--> statement-breakpoint
CREATE INDEX "research_cohort_created_by_idx" ON "research_cohorts" USING btree ("created_by");--> statement-breakpoint
CREATE INDEX "research_consent_patient_idx" ON "research_data_consent" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "research_consent_enabled_idx" ON "research_data_consent" USING btree ("consent_enabled");--> statement-breakpoint
CREATE INDEX "data_snapshot_study_idx" ON "research_data_snapshots" USING btree ("study_id");--> statement-breakpoint
CREATE INDEX "data_snapshot_created_idx" ON "research_data_snapshots" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "env_exposure_location_date_idx" ON "research_environmental_exposures" USING btree ("location_id","date");--> statement-breakpoint
CREATE INDEX "env_exposure_date_idx" ON "research_environmental_exposures" USING btree ("date");--> statement-breakpoint
CREATE INDEX "immune_marker_patient_idx" ON "research_immune_markers" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "immune_marker_name_idx" ON "research_immune_markers" USING btree ("marker_name");--> statement-breakpoint
CREATE INDEX "immune_marker_collection_idx" ON "research_immune_markers" USING btree ("collection_time");--> statement-breakpoint
CREATE INDEX "research_location_zipcode_idx" ON "research_locations" USING btree ("zip_code");--> statement-breakpoint
CREATE INDEX "research_location_coords_idx" ON "research_locations" USING btree ("latitude","longitude");--> statement-breakpoint
CREATE INDEX "research_measurement_visit_idx" ON "research_measurements" USING btree ("visit_id");--> statement-breakpoint
CREATE INDEX "research_measurement_patient_idx" ON "research_measurements" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "research_measurement_name_idx" ON "research_measurements" USING btree ("name");--> statement-breakpoint
CREATE INDEX "research_measurement_recorded_idx" ON "research_measurements" USING btree ("recorded_at");--> statement-breakpoint
CREATE INDEX "patient_location_patient_idx" ON "research_patient_locations" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "patient_location_location_idx" ON "research_patient_locations" USING btree ("location_id");--> statement-breakpoint
CREATE INDEX "patient_location_date_idx" ON "research_patient_locations" USING btree ("start_date","end_date");--> statement-breakpoint
CREATE INDEX "research_project_owner_idx" ON "research_projects" USING btree ("owner_id");--> statement-breakpoint
CREATE INDEX "research_project_status_idx" ON "research_projects" USING btree ("status");--> statement-breakpoint
CREATE INDEX "research_protocol_status_idx" ON "research_protocols" USING btree ("status");--> statement-breakpoint
CREATE INDEX "research_protocol_pi_idx" ON "research_protocols" USING btree ("principal_investigator");--> statement-breakpoint
CREATE INDEX "research_study_project_idx" ON "research_studies" USING btree ("project_id");--> statement-breakpoint
CREATE INDEX "research_study_status_idx" ON "research_studies" USING btree ("status");--> statement-breakpoint
CREATE INDEX "research_study_owner_idx" ON "research_studies" USING btree ("owner_user_id");--> statement-breakpoint
CREATE INDEX "research_visit_enrollment_idx" ON "research_visits" USING btree ("enrollment_id");--> statement-breakpoint
CREATE INDEX "research_visit_study_idx" ON "research_visits" USING btree ("study_id");--> statement-breakpoint
CREATE INDEX "research_visit_patient_idx" ON "research_visits" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "research_visit_scheduled_idx" ON "research_visits" USING btree ("scheduled_date");--> statement-breakpoint
CREATE INDEX "etl_job_type_idx" ON "risk_exposures_etl_jobs" USING btree ("job_type");--> statement-breakpoint
CREATE INDEX "etl_job_status_idx" ON "risk_exposures_etl_jobs" USING btree ("status");--> statement-breakpoint
CREATE INDEX "etl_job_created_idx" ON "risk_exposures_etl_jobs" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "robustness_protocol_idx" ON "robustness_reports" USING btree ("protocol_id");--> statement-breakpoint
CREATE INDEX "robustness_status_idx" ON "robustness_reports" USING btree ("overall_status");--> statement-breakpoint
CREATE INDEX "sentiment_analysis_patient_source_idx" ON "sentiment_analysis" USING btree ("patient_id","source_type");--> statement-breakpoint
CREATE INDEX "sentiment_analysis_polarity_idx" ON "sentiment_analysis" USING btree ("sentiment_polarity");--> statement-breakpoint
CREATE INDEX "session_metrics_session_idx" ON "session_metrics" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "study_enrollment_study_idx" ON "study_enrollments" USING btree ("study_id");--> statement-breakpoint
CREATE INDEX "study_enrollment_patient_idx" ON "study_enrollments" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "study_enrollment_status_idx" ON "study_enrollments" USING btree ("status");--> statement-breakpoint
CREATE INDEX "study_enrollment_unique_idx" ON "study_enrollments" USING btree ("study_id","patient_id");--> statement-breakpoint
CREATE INDEX "correlation_patient_symptom_idx" ON "symptom_environment_correlations" USING btree ("patient_id","symptom_type");--> statement-breakpoint
CREATE INDEX "correlation_significant_idx" ON "symptom_environment_correlations" USING btree ("is_statistically_significant");--> statement-breakpoint
CREATE INDEX "tool_exec_agent_idx" ON "tool_executions" USING btree ("agent_id");--> statement-breakpoint
CREATE INDEX "tool_exec_user_idx" ON "tool_executions" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "tool_exec_tool_idx" ON "tool_executions" USING btree ("tool_name");--> statement-breakpoint
CREATE INDEX "tool_exec_status_idx" ON "tool_executions" USING btree ("status");--> statement-breakpoint
CREATE INDEX "tool_exec_patient_idx" ON "tool_executions" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "tool_exec_created_idx" ON "tool_executions" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "trend_reports_user_period_idx" ON "trend_reports" USING btree ("user_id","period_start","period_end");--> statement-breakpoint
CREATE INDEX "presence_user_idx" ON "user_presence" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "presence_online_idx" ON "user_presence" USING btree ("is_online");--> statement-breakpoint
CREATE INDEX "vendor_accounts_user_vendor_idx" ON "vendor_accounts" USING btree ("user_id","vendor_id");--> statement-breakpoint
CREATE INDEX "vendor_accounts_status_idx" ON "vendor_accounts" USING btree ("connection_status");--> statement-breakpoint
CREATE INDEX "vendor_accounts_expiry_idx" ON "vendor_accounts" USING btree ("token_expires_at");--> statement-breakpoint
CREATE INDEX "video_exam_segments_session_idx" ON "video_exam_segments" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "video_exam_segments_exam_type_idx" ON "video_exam_segments" USING btree ("exam_type");--> statement-breakpoint
CREATE INDEX "video_exam_segments_status_idx" ON "video_exam_segments" USING btree ("status");--> statement-breakpoint
CREATE INDEX "video_exam_sessions_patient_idx" ON "video_exam_sessions" USING btree ("patient_id");--> statement-breakpoint
CREATE INDEX "video_exam_sessions_status_idx" ON "video_exam_sessions" USING btree ("status");--> statement-breakpoint
CREATE INDEX "video_exam_sessions_created_idx" ON "video_exam_sessions" USING btree ("created_at");--> statement-breakpoint
ALTER TABLE "chat_sessions" ADD CONSTRAINT "chat_sessions_context_patient_id_users_id_fk" FOREIGN KEY ("context_patient_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medications" ADD CONSTRAINT "medications_drug_id_drugs_id_fk" FOREIGN KEY ("drug_id") REFERENCES "public"."drugs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "medications" ADD CONSTRAINT "medications_prescribing_doctor_id_users_id_fk" FOREIGN KEY ("prescribing_doctor_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "drugs" ADD CONSTRAINT "drugs_rxcui_unique" UNIQUE("rxcui");--> statement-breakpoint
ALTER TABLE "patient_profiles" ADD CONSTRAINT "patient_profiles_followup_patient_id_unique" UNIQUE("followup_patient_id");
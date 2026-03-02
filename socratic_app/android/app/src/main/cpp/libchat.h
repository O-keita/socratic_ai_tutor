#ifndef LIBCHAT_H
#define LIBCHAT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chat_session chat_session;

/// Create a chat session — loads the GGUF model, creates context + sampler.
/// Returns NULL on failure.  CPU-only (ngl = 0).
chat_session * chat_create(const char * model_path, int n_ctx, int n_threads);

/// Generate a response for a user message.
/// Handles ChatML formatting and conversation history internally.
/// Returns a malloc'd string — caller must pass it to chat_string_free().
/// Returns NULL on error.
char * chat_generate(chat_session * session, const char * user_message);

/// Free a string returned by chat_generate().
void chat_string_free(char * str);

/// Reset the conversation — clears message history and KV cache.
/// The model stays loaded, so the next chat_generate() starts a fresh conversation.
void chat_reset(chat_session * session);

/// Destroy the session — frees model, context, sampler, and message history.
void chat_destroy(chat_session * session);

#ifdef __cplusplus
}
#endif

#endif // LIBCHAT_H

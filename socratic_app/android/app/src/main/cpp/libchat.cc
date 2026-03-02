#include "libchat.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define TAG "libchat"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) fprintf(stdout, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif

struct chat_session {
    llama_model   * model   = nullptr;
    llama_context * ctx     = nullptr;
    llama_sampler * sampler = nullptr;
    const llama_vocab * vocab = nullptr;

    // Conversation history for chat template formatting.
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted;
    int prev_len = 0;
};

// ---------------------------------------------------------------------------
// chat_create
// ---------------------------------------------------------------------------

extern "C" chat_session * chat_create(
    const char * model_path,
    int n_ctx,
    int n_threads
) {
    // Only log errors from llama.cpp.
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            LOGE("%s", text);
        }
    }, nullptr);

    ggml_backend_load_all();

    // ---- model ----
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU-only

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        LOGE("chat_create: failed to load model from %s", model_path);
        return nullptr;
    }

    // ---- context ----
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_ctx;
    ctx_params.n_batch = n_ctx;
    if (n_threads > 0) {
        ctx_params.n_threads       = n_threads;
        ctx_params.n_threads_batch = n_threads;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("chat_create: failed to create context");
        llama_model_free(model);
        return nullptr;
    }

    // ---- sampler ----
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.6f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ---- session ----
    auto * session    = new chat_session();
    session->model    = model;
    session->ctx      = ctx;
    session->sampler  = smpl;
    session->vocab    = llama_model_get_vocab(model);
    session->formatted.resize(n_ctx);
    session->prev_len = 0;

    LOGI("chat_create: model loaded, ctx=%d threads=%d", n_ctx, n_threads);
    return session;
}

// ---------------------------------------------------------------------------
// chat_generate  (non-streaming, returns full response)
// ---------------------------------------------------------------------------

extern "C" char * chat_generate(
    chat_session * session,
    const char * user_message
) {
    if (!session || !user_message) return nullptr;

    // ---- append user message ----
    session->messages.push_back({"user", strdup(user_message)});

    // ---- apply chat template ----
    const char * tmpl = llama_model_chat_template(session->model, nullptr);

    int new_len = llama_chat_apply_template(
        tmpl,
        session->messages.data(),
        session->messages.size(),
        true,
        session->formatted.data(),
        session->formatted.size()
    );
    if (new_len > (int)session->formatted.size()) {
        session->formatted.resize(new_len);
        new_len = llama_chat_apply_template(
            tmpl,
            session->messages.data(),
            session->messages.size(),
            true,
            session->formatted.data(),
            session->formatted.size()
        );
    }
    if (new_len < 0) {
        LOGE("chat_generate: failed to apply chat template");
        return nullptr;
    }

    // The prompt is the delta since the last call.
    std::string prompt(
        session->formatted.begin() + session->prev_len,
        session->formatted.begin() + new_len
    );

    // ---- tokenize ----
    const bool is_first =
        llama_memory_seq_pos_max(llama_get_memory(session->ctx), 0) == -1;

    const int n_prompt_tokens = -llama_tokenize(
        session->vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true
    );
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(
            session->vocab, prompt.c_str(), prompt.size(),
            prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        LOGE("chat_generate: tokenization failed");
        return nullptr;
    }

    // ---- decode loop ----
    std::string response;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    while (true) {
        int ctx_used = llama_memory_seq_pos_max(llama_get_memory(session->ctx), 0) + 1;
        if (ctx_used + batch.n_tokens > llama_n_ctx(session->ctx)) {
            LOGE("chat_generate: context size exceeded");
            break;
        }

        if (llama_decode(session->ctx, batch) != 0) {
            LOGE("chat_generate: decode failed");
            break;
        }

        llama_token new_token = llama_sampler_sample(session->sampler, session->ctx, -1);

        if (llama_vocab_is_eog(session->vocab, new_token)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(session->vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            response.append(buf, n);
        }

        batch = llama_batch_get_one(&new_token, 1);
    }

    // ---- record assistant response in history ----
    session->messages.push_back({"assistant", strdup(response.c_str())});
    session->prev_len = llama_chat_apply_template(
        tmpl,
        session->messages.data(),
        session->messages.size(),
        false,
        nullptr,
        0
    );

    LOGI("chat_generate: response length=%zu", response.size());

    // Return a malloc'd copy the caller can free.
    char * result = (char *)malloc(response.size() + 1);
    if (result) {
        memcpy(result, response.c_str(), response.size() + 1);
    }
    return result;
}

// ---------------------------------------------------------------------------
// chat_reset  (clear conversation, keep model loaded)
// ---------------------------------------------------------------------------

extern "C" void chat_reset(chat_session * session) {
    if (!session) return;

    // Free strdup'd message content strings.
    for (auto & msg : session->messages) {
        free(const_cast<char *>(msg.content));
    }
    session->messages.clear();
    session->prev_len = 0;

    // Clear the KV cache so the context starts fresh.
    llama_memory_clear(llama_get_memory(session->ctx), true);

    LOGI("chat_reset: conversation cleared, model still loaded");
}

// ---------------------------------------------------------------------------
// chat_string_free
// ---------------------------------------------------------------------------

extern "C" void chat_string_free(char * str) {
    free(str);
}

// ---------------------------------------------------------------------------
// chat_destroy
// ---------------------------------------------------------------------------

extern "C" void chat_destroy(chat_session * session) {
    if (!session) return;

    for (auto & msg : session->messages) {
        free(const_cast<char *>(msg.content));
    }
    session->messages.clear();

    if (session->sampler) llama_sampler_free(session->sampler);
    if (session->ctx)     llama_free(session->ctx);
    if (session->model)   llama_model_free(session->model);

    delete session;
    LOGI("chat_destroy: session freed");
}

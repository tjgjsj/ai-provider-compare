"""
AI Model Comparison Tool
Compare multiple OpenAI models side-by-side
Built by Tony | Portfolio Project
"""

import streamlit as st
import time
from openai import OpenAI

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Model Comparison Tool",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stMarkdown h1 { color: #1a1a2e; }
    div[data-testid="stHorizontalBlock"] > div {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Configurations
# ─────────────────────────────────────────────
OPENAI_MODELS = {
    "GPT-4o Mini (cheapest)": {
        "id": "gpt-4o-mini",
        "input_cost": 0.15,   # per 1M tokens
        "output_cost": 0.60,
        "description": "Fast & cheap, great for most tasks",
    },
    "GPT-4o": {
        "id": "gpt-4o",
        "input_cost": 2.50,
        "output_cost": 10.00,
        "description": "Flagship model, best quality",
    },
    "GPT-4.1 Nano (newest cheap)": {
        "id": "gpt-4.1-nano",
        "input_cost": 0.10,
        "output_cost": 0.40,
        "description": "Newest generation, ultra-cheap",
    },
    "GPT-4.1 Mini": {
        "id": "gpt-4.1-mini",
        "input_cost": 0.40,
        "output_cost": 1.60,
        "description": "Newest generation, balanced",
    },
}


def estimate_cost(input_tokens: int, output_tokens: int, model_config: dict) -> float:
    """Estimate cost in dollars based on token usage and model pricing."""
    input_cost = (input_tokens / 1_000_000) * model_config["input_cost"]
    output_cost = (output_tokens / 1_000_000) * model_config["output_cost"]
    return input_cost + output_cost


# ─────────────────────────────────────────────
# API Call Function
# ─────────────────────────────────────────────
def call_openai(prompt: str, system_prompt: str, model_config: dict, max_tokens: int) -> dict:
    """Call OpenAI API and return response with metrics."""
    try:
        client = OpenAI(api_key=st.session_state.openai_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        response = client.chat.completions.create(
            model=model_config["id"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=st.session_state.temperature,
        )
        elapsed = time.time() - start_time

        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = estimate_cost(input_tokens, output_tokens, model_config)

        return {
            "success": True,
            "content": content,
            "response_time": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost": cost,
            "model_used": model_config["id"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_result(result: dict, model_name: str):
    """Render a single model's result in a column."""
    if result["success"]:
        st.markdown(result["content"])
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("⏱️ Time", f"{result['response_time']:.2f}s")
        m2.metric("📊 Tokens", f"{result['total_tokens']:,}")
        m3.metric("💰 Cost", f"${result['estimated_cost']:.6f}")
        with st.expander("Details"):
            st.json({
                "model": result["model_used"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "estimated_cost_usd": f"${result['estimated_cost']:.6f}",
            })
    else:
        st.error(f"❌ {result['error']}")


# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────
st.title("⚡ AI Model Comparison Tool")
st.markdown("*Compare multiple OpenAI models side-by-side — response quality, speed, and cost*")

# ─── Sidebar: Configuration ───
with st.sidebar:
    st.header("🔑 API Configuration")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key — never stored on disk"
    )
    if openai_key:
        st.session_state.openai_key = openai_key

    st.divider()

    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
        help="Lower = more focused, Higher = more creative")
    st.session_state.temperature = temperature

    max_tokens = st.slider("Max Output Tokens", 100, 2000, 500, 100)

    st.divider()

    st.header("📊 Compare Mode")
    compare_mode = st.radio(
        "How many models?",
        ["2 Models (side-by-side)", "3 Models (triple compare)"],
        help="Compare 2 or 3 models at once"
    )
    num_models = 2 if "2 Models" in compare_mode else 3

    st.divider()
    st.markdown("---")
    st.markdown("**💰 Cost Guide (per 1M tokens):**")
    for name, config in OPENAI_MODELS.items():
        st.caption(f"**{name}**: ${config['input_cost']} in / ${config['output_cost']} out")


# ─── Model Selection Row ───
model_names = list(OPENAI_MODELS.keys())

if num_models == 2:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### 🔵 Model A")
        model_a_name = st.selectbox("Model A", model_names, index=0, label_visibility="collapsed")
        st.caption(OPENAI_MODELS[model_a_name]["description"])
    with col_m2:
        st.markdown("### 🟢 Model B")
        model_b_name = st.selectbox("Model B", model_names, index=1, label_visibility="collapsed")
        st.caption(OPENAI_MODELS[model_b_name]["description"])
    selected_models = [(model_a_name, "🔵"), (model_b_name, "🟢")]
else:
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("### 🔵 Model A")
        model_a_name = st.selectbox("Model A", model_names, index=0, label_visibility="collapsed")
        st.caption(OPENAI_MODELS[model_a_name]["description"])
    with col_m2:
        st.markdown("### 🟢 Model B")
        model_b_name = st.selectbox("Model B", model_names, index=2, label_visibility="collapsed")
        st.caption(OPENAI_MODELS[model_b_name]["description"])
    with col_m3:
        st.markdown("### 🟠 Model C")
        model_c_name = st.selectbox("Model C", model_names, index=1, label_visibility="collapsed")
        st.caption(OPENAI_MODELS[model_c_name]["description"])
    selected_models = [(model_a_name, "🔵"), (model_b_name, "🟢"), (model_c_name, "🟠")]

# ─── System Prompt (Collapsible) ───
with st.expander("🛠️ System Prompt (Optional)"):
    system_prompt = st.text_area(
        "System prompt applied to all models",
        placeholder="e.g., You are a helpful assistant that responds concisely.",
        height=80,
        label_visibility="collapsed",
    )

# ─── Prompt Input ───
st.markdown("---")
prompt = st.text_area(
    "💬 Enter your prompt",
    placeholder="Ask anything! All selected models will answer the same question.",
    height=120,
)

# ─── Sample Prompts ───
st.markdown("**Quick prompts:**")
sample_cols = st.columns(4)
samples = [
    "Explain microservices vs monolith in 3 sentences",
    "Write a Python function to validate email addresses",
    "What are 3 creative uses for AI in firefighting?",
    "Compare SQL and NoSQL databases",
]
for i, sample in enumerate(samples):
    with sample_cols[i]:
        if st.button(sample[:30] + "...", key=f"sample_{i}", use_container_width=True):
            st.session_state.sample_prompt = sample
            st.rerun()

# Use sample prompt if one was clicked
if "sample_prompt" in st.session_state:
    prompt = st.session_state.sample_prompt
    del st.session_state.sample_prompt

# ─── Compare Button ───
st.markdown("")
compare_clicked = st.button(
    "🚀 Compare Responses",
    type="primary",
    use_container_width=True,
    disabled=not prompt,
)

# ─── Results ───
if compare_clicked and prompt:
    if not openai_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # Call all selected models
    results = []
    columns = st.columns(num_models)

    for i, (model_name, icon) in enumerate(selected_models):
        config = OPENAI_MODELS[model_name]
        with columns[i]:
            st.markdown(f"### {icon} {config['id']}")
            with st.spinner(f"Calling {config['id']}..."):
                result = call_openai(prompt, system_prompt, config, max_tokens)
                results.append((model_name, config, result))

    # Display results
    for i, (model_name, config, result) in enumerate(results):
        with columns[i]:
            render_result(result, model_name)

    # ─── Comparison Summary ───
    successful = [(name, cfg, r) for name, cfg, r in results if r["success"]]

    if len(successful) >= 2:
        st.markdown("---")
        st.markdown("### 📊 Head-to-Head Comparison")

        comp_cols = st.columns(3)

        with comp_cols[0]:
            fastest = min(successful, key=lambda x: x[2]["response_time"])
            slowest = max(successful, key=lambda x: x[2]["response_time"])
            diff = slowest[2]["response_time"] - fastest[2]["response_time"]
            st.metric("⚡ Fastest", fastest[1]["id"],
                f"{diff:.2f}s faster than {slowest[1]['id']}")

        with comp_cols[1]:
            most_concise = min(successful, key=lambda x: x[2]["output_tokens"])
            most_verbose = max(successful, key=lambda x: x[2]["output_tokens"])
            token_diff = most_verbose[2]["output_tokens"] - most_concise[2]["output_tokens"]
            st.metric("📝 Most Concise", most_concise[1]["id"],
                f"{token_diff:,} fewer tokens than {most_verbose[1]['id']}")

        with comp_cols[2]:
            cheapest = min(successful, key=lambda x: x[2]["estimated_cost"])
            priciest = max(successful, key=lambda x: x[2]["estimated_cost"])
            cost_diff = priciest[2]["estimated_cost"] - cheapest[2]["estimated_cost"]
            st.metric("💰 Cheapest", cheapest[1]["id"],
                f"${cost_diff:.6f} saved vs {priciest[1]['id']}")

        # ─── Detailed Comparison Table ───
        st.markdown("### 📋 Full Breakdown")
        table_data = {
            "Model": [], "Response Time (s)": [], "Input Tokens": [],
            "Output Tokens": [], "Total Tokens": [], "Est. Cost ($)": [],
        }
        for name, cfg, result in successful:
            table_data["Model"].append(cfg["id"])
            table_data["Response Time (s)"].append(f"{result['response_time']:.2f}")
            table_data["Input Tokens"].append(f"{result['input_tokens']:,}")
            table_data["Output Tokens"].append(f"{result['output_tokens']:,}")
            table_data["Total Tokens"].append(f"{result['total_tokens']:,}")
            table_data["Est. Cost ($)"].append(f"${result['estimated_cost']:.6f}")

        st.table(table_data)

        total_cost = sum(r[2]["estimated_cost"] for r in successful)
        total_tokens = sum(r[2]["total_tokens"] for r in successful)
        st.info(
            f"**This comparison used {total_tokens:,} total tokens "
            f"and cost approximately ${total_cost:.6f}**"
        )

# ─── Session Running Total ───
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
    st.session_state.session_comparisons = 0

if compare_clicked and prompt:
    successful_results = [r for _, _, r in results if r["success"]]
    if successful_results:
        st.session_state.session_cost += sum(r["estimated_cost"] for r in successful_results)
        st.session_state.session_comparisons += 1

# ─── Footer ───
st.markdown("---")
footer_cols = st.columns([3, 1])
with footer_cols[0]:
    st.caption("Built with Streamlit + OpenAI API | Portfolio Project by Tony")
with footer_cols[1]:
    if st.session_state.session_comparisons > 0:
        st.caption(
            f"Session: {st.session_state.session_comparisons} comparisons | "
            f"${st.session_state.session_cost:.6f} total"
        )

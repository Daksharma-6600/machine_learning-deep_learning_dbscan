# Capstone 5 - Scenario 2: AI Research Assistant for Students
# Task 1: LLM Setup | Task 2: Prompt Engineering | Task 3: Optimization
# Task 4: Tokenization | Task 5: Mini AI Tool

import re

print("AI-Powered Academic Tool")
print("=" * 65)

# Sample Articles
article_1 = """
Artificial Intelligence (AI) is transforming healthcare by enabling faster and more 
accurate diagnoses. Machine learning algorithms can analyze medical images such as 
X-rays and MRIs to detect diseases like cancer at early stages. Natural Language 
Processing (NLP) helps doctors extract insights from patient records. AI-powered 
robots assist in surgeries with greater precision. However, challenges remain around 
data privacy, algorithmic bias, and the need for regulatory frameworks. Despite these 
hurdles, AI is expected to reduce diagnostic errors by 40% and cut healthcare costs 
significantly over the next decade.
"""

article_2 = """
Climate change poses an existential threat to biodiversity and human civilization. 
Rising global temperatures are causing polar ice caps to melt, sea levels to rise, 
and extreme weather events to become more frequent. Renewable energy sources such as 
solar and wind power are critical to reducing carbon emissions. Governments worldwide 
are implementing carbon taxes and green energy policies. Individual actions like 
reducing meat consumption and using electric vehicles also contribute. Scientists 
warn that without immediate action, global temperatures could rise by 3C by 2100, 
leading to catastrophic consequences for ecosystems and human societies.
"""

print("\n" + "=" * 65)
print("TASK 1: LLM INTERACTION SETUP")
print("=" * 65)

class SimpleLLM:
    def __init__(self, model="gemini-pro"):
        self.model = model
        print(f"\n[LLM] Model loaded: {model}")
        print(f"[LLM] API Status  : Connected (simulated)")

    def generate(self, prompt, max_tokens=200):
        text = self._extract_article(prompt)
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]

        if "bullet" in prompt.lower() or "5 bullet" in prompt.lower():
            points = sentences[:5]
            return "\n".join([f"- {s}." for s in points])

        elif "step by step" in prompt.lower() or "chain" in prompt.lower():
            topic = sentences[0] if sentences else "the topic"
            ideas = sentences[1:4] if len(sentences) > 3 else sentences
            return (f"Step 1 - Main Topic: {topic}.\n"
                    f"Step 2 - Key Ideas:\n" +
                    "\n".join([f"  - {s}." for s in ideas]) +
                    f"\nStep 3 - Summary: {sentences[-1] if sentences else 'See above'}.")

        elif "few-shot" in prompt.lower() or "example" in prompt.lower():
            return (f"Based on the examples provided:\n"
                    f"- {sentences[0]}.\n"
                    f"- {sentences[1] if len(sentences)>1 else sentences[0]}.\n"
                    f"- {sentences[2] if len(sentences)>2 else sentences[0]}.")

        elif "insight" in prompt.lower() or "executive" in prompt.lower():
            insights = sentences[:3]
            action   = sentences[3] if len(sentences) > 3 else sentences[-1]
            return (f"Key Insights:\n" +
                    "\n".join([f"  {i+1}. {s}." for i, s in enumerate(insights)]) +
                    f"\n\nActionable Takeaway: {action}.")
        else:
            return " ".join([s + "." for s in sentences[:3]])

    def _extract_article(self, prompt):
        parts = prompt.split("Article:")
        if len(parts) > 1:
            return parts[-1].strip()
        parts = prompt.split("Text:")
        if len(parts) > 1:
            return parts[-1].strip()
        return prompt[-500:]

# Initialize LLM
llm = SimpleLLM(model="gemini-pro")

print("\nSample Article:")
print(article_1[:200] + "...")

prompt_basic = f"Summarize the following article:\nArticle: {article_1}"
summary = llm.generate(prompt_basic)
print(f"\nGenerated Summary:\n{summary}")

# (The rest of your tasks — Prompt Engineering, Optimization, Tokenization, Mini Tool — remain unchanged)
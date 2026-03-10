import os
import gradio as gr
from google import genai
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the FAISS index and saved data files
_index = faiss.read_index('reviews.index')
with open('chunks.pkl', 'rb') as f:
    _chunks = pickle.load(f)
with open('metadata.pkl', 'rb') as f:
    _metadata = pickle.load(f)

# Load the sentence embedding model
_embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print(f'Index: {_index.ntotal} vectors | Chunks: {len(_chunks)} | Metadata: {len(_metadata)}')


def translate_to_english(query, client):
    """Translate any query to English so it matches the English review index."""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=(
            f'Translate the following search query to English. '
            f'Return only the translated text, nothing else.\n\nQuery: {query}'
        )
    )
    return response.text.strip()


def search_reviews(query, score_filter, product_filter, api_key, top_k):
    """Search reviews using FAISS and generate a Turkish summary with Gemini."""

    if not query.strip():
        return '⚠️ Lütfen bir sorgu girin.', ''

    # Create the Gemini client once and reuse it for both translation and summary
    client = genai.Client(api_key=api_key.strip()) if api_key.strip() else None

    # Translate the query to English before searching,
    # because the review index contains only English text
    search_query = query
    if client:
        try:
            search_query = translate_to_english(query, client)
        except Exception:
            pass  # If translation fails, use the original query

    # Convert the query text into a vector
    q_vec = _embed_model.encode([search_query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)  # Normalize for cosine similarity

    # Search the top 300 candidates, then apply filters
    distances, indices = _index.search(q_vec, min(300, _index.ntotal))

    filtered = []
    for idx, sim in zip(indices[0], distances[0]):
        meta = _metadata[idx]

        # Skip if score does not match the filter
        if score_filter != 'Tümü' and meta['Score'] != int(score_filter[0]):
            continue

        # Skip if product ID does not match the filter
        pid = product_filter.strip()
        if pid and meta['ProductId'] != pid:
            continue

        filtered.append((int(idx), float(sim), meta))

        # Stop when we have enough results
        if len(filtered) >= int(top_k):
            break

    if not filtered:
        return '❌ Filtrelere uyan sonuç bulunamadı.', ''

    # Join selected chunks as context for the LLM
    context = '\n\n---\n\n'.join(_chunks[idx] for idx, _, _ in filtered)

    # Send context to Gemini and ask for a Turkish summary
    if client:
        try:
            prompt = (
                f'Sen bir Amazon ürün yorumu analistsin.\n'
                f'Kullanıcı sorusu: {query}\n\n'
                f'İlgili yorumlar:\n{context}\n\n'
                f'Bu yorumlara dayanarak soruyu Türkçe olarak yanıtla. '
                f'3-5 cümlelik kısa ve net bir özet yaz. '
                f'Öne çıkan artıları, eksileri ve genel kullanıcı görüşünü belirt.'
            )
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            llm_answer = response.text
        except Exception as e:
            llm_answer = f'⚠️ Gemini hatası: {e}'
    else:
        llm_answer = '⚠️ Gemini API key giriniz (sol panelde).'

    # Build review cards for display
    cards = ''
    for rank, (idx, sim, meta) in enumerate(filtered, 1):
        stars = '★' * meta['Score'] + '☆' * (5 - meta['Score'])
        cards += f'### [{rank}] {stars} &nbsp; Benzerlik: {sim:.3f} &nbsp; `{meta["ProductId"]}`\n\n'
        cards += _chunks[idx] + '\n\n---\n\n'

    return llm_answer, cards


# === Gradio UI ===
with gr.Blocks(title='Amazon Review RAG') as demo:
    gr.Markdown('# 🛒 Amazon Ürün Yorumu — RAG Arama')
    gr.Markdown(
        'FAISS vektör araması + Gemini LLM ile 10.000 Amazon yorumunu sorgula. '
        'Puan veya Ürün ID ile filtrele.'
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            api_key_box = gr.Textbox(
                label='🔑 Gemini API Key',
                placeholder='AIza...',
                type='password',
                value=os.environ.get('GEMINI_API_KEY', '')
            )
            query_box = gr.Textbox(
                label='🔍 Sorgu',
                placeholder='Örn: köpek maması kaliteli mi?',
                lines=3
            )
            score_dd = gr.Dropdown(
                choices=['Tümü', '5★', '4★', '3★', '2★', '1★'],
                value='Tümü',
                label='⭐ Puan Filtresi'
            )
            product_box = gr.Textbox(
                label='📦 Ürün ID (opsiyonel)',
                placeholder='Örn: B001E4KFG0'
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label='Gösterilecek Sonuç Sayısı'
            )
            search_btn = gr.Button('🔍 Ara', variant='primary', size='lg')

        with gr.Column(scale=2):
            with gr.Tab('🤖 LLM Özeti'):
                llm_out = gr.Markdown()
            with gr.Tab('📋 Bulunan Yorumlar'):
                reviews_out = gr.Markdown()

    # Example queries to help users get started
    gr.Examples(
        examples=[
            ['best dog food for large breeds', 'Tümü', '', 5],
            ['terrible taste not recommended', '1★', '', 5],
            ['organic coffee with great aroma', '5★', '', 5],
            ['healthy snack for kids', 'Tümü', '', 5],
        ],
        inputs=[query_box, score_dd, product_box, top_k_slider],
        label='Örnek Sorgular'
    )

    # Connect the search button to the search function
    search_btn.click(
        fn=search_reviews,
        inputs=[query_box, score_dd, product_box, api_key_box, top_k_slider],
        outputs=[llm_out, reviews_out]
    )

demo.launch(
    share=False,
    server_name='127.0.0.1',
    server_port=7860,
    theme=gr.themes.Soft()
)

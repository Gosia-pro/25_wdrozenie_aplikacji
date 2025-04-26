import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from dotenv import load_dotenv
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from io import BytesIO
import os
from dotenv import dotenv_values

#print(base64_audio)  # Check the content of the base64 string

import base64

#audio_bytes = audio.getvalue()
#base64_audio = base64.b64encode(audio_bytes).decode('utf-8')


# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

env = dotenv_values(".env")
#https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if "QDRANT_URL" in st.secrets:
    env["QDRANT_URL"] = st.secrets["QDRANT_URL"]
if "QDRANT_API_KEY" in st.secrets:
    env["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
    
   
# Funkcja do pobierania sekretów z .env lub streamlit.secrets
def get_secret(key: str) -> str:
    """
    Pobiera wartość sekretu:
    - Najpierw próbuje ze streamlit.secrets
    - Jeśli nie znajdzie, bierze z os.environ (czyli .env)
    """
    if key in st.secrets:
        return st.secrets[key]
    elif key in os.environ:
        return os.getenv(key)
    else:
        raise ValueError(f"Secret '{key}' not found in Streamlit secrets or .env environment variables.")

# Konfiguracja API
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
QDRANT_COLLECTION_NAME = "notes"

# Klient OpenAI
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

# Transkrypcja audio
def transcribe_audio(audio_bytes):
    try:
        openai_client = get_openai_client()
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model=AUDIO_TRANSCRIBE_MODEL,
            response_format="verbose_json",
        )
        return transcript.text
    except Exception as e:
        st.error(f"Nie ud ało się przetworzyć audio: {e}")
        return ""

# Qdrant Client
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),  # Pobieramy URL z pliku .env
        api_key=os.getenv("QDRANT_API_KEY")  # Pobieramy klucz API z pliku .env
    )

# Sprawdzenie istnienia kolekcji w Qdrant
def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")

# Tworzenie embeddingów
def get_embeddings(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding

# Dodawanie notatki do bazy danych
def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[PointStruct(
            id=points_count.count + 1,
            vector=get_embeddings(text=note_text),
            payload={"text": note_text},
        )]
    )

# Wyszukiwanie notatek z bazy danych
def list_notes_from_db(query=None, offset=0, limit=10):
    qdrant_client = get_qdrant_client()
    if not query:
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=limit, offset=offset)[0]
    else:
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embeddings(text=query),
            limit=limit,
            offset=offset,
        )
    result = [{"text": note.payload["text"], "score": note.score} for note in notes]
    return result

# Główna część aplikacji Streamlit
st.set_page_config(page_title="Audio Notatki", layout="centered")

# Sprawdzanie klucza API OpenAI
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# Inicjalizacja sesji
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

st.title("Audio Notatki")
assure_db_collection_exists()

add_tab, search_tab = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])

with add_tab:
    note_audio = audiorecorder(
        start_prompt="Nagraj notatkę",
        stop_prompt="Zatrzymaj nagrywanie",
    )
    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

        if st.button("Transkrybuj audio"):
            st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])

        if st.session_state["note_audio_text"]:
            st.session_state["note_text"] = st.text_area("Edytuj notatkę", value=st.session_state["note_audio_text"])

        if st.session_state["note_text"] and st.button("Zapisz notatkę", disabled=not st.session_state["note_text"]):
            add_note_to_db(note_text=st.session_state["note_text"])
            st.toast("Notatka zapisana", icon="🎉")

with search_tab:
    query = st.text_input("Wyszukaj notatkę")
    if st.button("Szukaj"):
        for note in list_notes_from_db(query):
            with st.container(border=True):
                st.markdown(note["text"])
                if note["score"]:
                    st.markdown(f':violet[{note["score"]}]')

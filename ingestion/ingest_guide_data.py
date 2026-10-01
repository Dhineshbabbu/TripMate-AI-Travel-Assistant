import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" 

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "tripmate-destinations"
)

BATCH_SIZE = 50

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIMENSION = 384

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    exist_ok=True
)

LOG_FILE = LOG_DIR / "ingestion.log"


logger = logging.getLogger("tripmate_ingestion")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Console handler
console_handler = logging.StreamHandler()

console_handler.setFormatter(
    formatter
)


# File handler
file_handler = logging.FileHandler(
    LOG_FILE,
    encoding="utf-8"
)

file_handler.setFormatter(
    formatter
)


logger.addHandler(
    console_handler
)

logger.addHandler(
    file_handler
)



if not PINECONE_API_KEY:

    logger.error(
        "PINECONE_API_KEY is missing."
    )

    raise ValueError(
        "PINECONE_API_KEY is not configured "
        "in the .env file."
    )


if not DATA_DIR.exists():

    logger.error(
        "Destination data directory does not exist: %s",
        DATA_DIR
    )

    raise FileNotFoundError(
        f"Destination directory not found: {DATA_DIR}"
    )


logger.info(
    "Loading embedding model: %s",
    EMBEDDING_MODEL
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

logger.info(
    "Embedding model loaded successfully."
)



SECTION_NAMES = [
    "VISA & ENTRY",
    "BEST TIME TO VISIT",
    "LOCAL CUSTOMS",
    "PACKING TIPS",
    "SAFETY & HEALTH",
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
)


def parse_destination_file(
    file_path: Path
) -> list[Document]:

    logger.info(
        "Reading destination file: %s",
        file_path.name
    )

    text = file_path.read_text(
        encoding="utf-8"
    )

    city = file_path.stem.replace(
        "_",
        " "
    ).title()

    logger.info(
        "Detected destination: %s",
        city
    )

    documents = []

    current_section = None

    current_content = []

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ------------------------------------------
        # Detect section heading
        # ------------------------------------------

        if line in SECTION_NAMES:

            # Save previous section
            if (
                current_section
                and current_content
            ):

                documents.extend(
                    create_section_documents(
                        city=city,
                        source=file_path.name,
                        section=current_section,
                        content="\n".join(
                            current_content
                        ),
                    )
                )

            current_section = line

            current_content = []

        elif current_section:

            current_content.append(line)

    # ------------------------------------------
    # Save final section
    # ------------------------------------------

    if (
        current_section
        and current_content
    ):

        documents.extend(
            create_section_documents(
                city=city,
                source=file_path.name,
                section=current_section,
                content="\n".join(
                    current_content
                ),
            )
        )

    logger.info(
        "Created %d chunks from %s",
        len(documents),
        file_path.name
    )

    return documents


# ============================================================
# CREATE SECTION DOCUMENTS
# ============================================================

def create_section_documents(
    city: str,
    source: str,
    section: str,
    content: str,
) -> list[Document]:

    base_document = Document(
        page_content=content,
        metadata={
            "city": city,
            "topic": section.lower(),
            "source": source,
        },
    )

    chunks = text_splitter.split_documents(
        [base_document]
    )

    for chunk_id, chunk in enumerate(chunks):

        chunk.metadata[
            "chunk_id"
        ] = chunk_id

    logger.info(
        "Section '%s' for %s → %d chunk(s)",
        section,
        city,
        len(chunks)
    )

    return chunks


# ============================================================
# CREATE PINECONE INDEX
# ============================================================

def initialize_pinecone():

    logger.info(
        "Connecting to Pinecone..."
    )

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    existing_indexes = [
        index.name
        for index in pc.list_indexes()
    ]

    if INDEX_NAME not in existing_indexes:

        logger.info(
            "Creating Pinecone index: %s",
            INDEX_NAME
        )

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        logger.info(
            "Pinecone index created successfully."
        )

    else:

        logger.info(
            "Using existing Pinecone index: %s",
            INDEX_NAME
        )

    return pc.Index(INDEX_NAME)


# ============================================================
# DOCUMENT ID
# ============================================================

def create_document_id(
    document: Document
) -> str:

    city = (
        document.metadata["city"]
        .lower()
        .replace(" ", "_")
    )

    topic = (
        document.metadata["topic"]
        .lower()
        .replace(" ", "_")
    )

    source = document.metadata[
        "source"
    ]

    chunk_id = document.metadata[
        "chunk_id"
    ]

    return (
        f"{city}__"
        f"{topic}__"
        f"{source}__"
        f"{chunk_id}"
    )


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

def load_all_documents():

    logger.info(
        "Searching destination directory: %s",
        DATA_DIR
    )

    files = sorted(
        DATA_DIR.glob("*.txt")
    )

    if not files:

        logger.error(
            "No .txt destination files found."
        )

        raise FileNotFoundError(
            "No destination .txt files found."
        )

    logger.info(
        "Found %d destination file(s).",
        len(files)
    )

    all_documents = []

    for file_path in files:

        try:

            documents = parse_destination_file(
                file_path
            )

            all_documents.extend(
                documents
            )

        except Exception:

            logger.exception(
                "Failed to process file: %s",
                file_path.name
            )

            raise

    logger.info(
        "Total documents/chunks created: %d",
        len(all_documents)
    )

    return all_documents


# ============================================================
# BATCH INGESTION
# ============================================================

def ingest_documents(
    vector_store,
    documents: list[Document]
):

    total_documents = len(documents)

    logger.info(
        "Starting batch ingestion."
    )

    logger.info(
        "Total chunks: %d | Batch size: %d",
        total_documents,
        BATCH_SIZE
    )

    for start in range(
        0,
        total_documents,
        BATCH_SIZE
    ):

        batch = documents[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        logger.info(
            "Processing batch %d | "
            "chunks %d-%d",
            batch_number,
            start + 1,
            start + len(batch)
        )

        try:

            ids = [
                create_document_id(
                    document
                )
                for document in batch
            ]

            vector_store.add_documents(
                documents=batch,
                ids=ids,
            )

            logger.info(
                "Batch %d successfully inserted.",
                batch_number
            )

        except Exception:

            logger.exception(
                "Batch %d ingestion failed.",
                batch_number
            )

            raise

    logger.info(
        "All batches successfully ingested."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "========================================"
    )

    logger.info(
        "TripMate RAG ingestion started."
    )

    logger.info(
        "========================================"
    )

    try:

        # --------------------------------------
        # Load documents
        # --------------------------------------

        documents = load_all_documents()

        # --------------------------------------
        # Initialize Pinecone
        # --------------------------------------

        index = initialize_pinecone()

        # --------------------------------------
        # Create LangChain vector store
        # --------------------------------------

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
        )

        # --------------------------------------
        # Batch ingestion
        # --------------------------------------

        ingest_documents(
            vector_store=vector_store,
            documents=documents,
        )

        logger.info(
            "========================================"
        )

        logger.info(
            "TripMate RAG ingestion completed."
        )

        logger.info(
            "Index: %s",
            INDEX_NAME
        )

        logger.info(
            "Total chunks: %d",
            len(documents)
        )

        logger.info(
            "Log file: %s",
            LOG_FILE
        )

        logger.info(
            "========================================"
        )

    except Exception:

        logger.exception(
            "TripMate RAG ingestion failed."
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


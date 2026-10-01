import os
from pathlib import Path
from datetime import datetime
import calendar
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "tripmate-destinations"
)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pinecone_vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding_model)


retriever = pinecone_vector_store.as_retriever(search_kwargs={"k": 3})

@tool(
    description=(
        "Search the TripMate destination knowledge base in Pinecone. "
        "Use this tool for destination information including visa, "
        "best time to visit, local customs, packing tips, safety, "
        "and health."
    )
)
def search_destination_guide(query: str) -> str:

    try:
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant destination information was found."

        results = []

        for doc in docs:
            metadata = doc.metadata

            city = metadata.get("city", "Unknown")
            topic = metadata.get("topic", "Unknown")

            results.append(
                f"City: {city}\n"
                f"Topic: {topic}\n"
                f"Information: {doc.page_content}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Destination retrieval failed: {str(e)}"


@tool(
    description=(
        "Get weather information for a location using latitude, longitude, "
        "and month. Returns the temperature range and weather conditions "
        "for that month."
    )
)
def get_weather_forecast(latitude: float,longitude: float,month: str) -> dict:

    try:
        # Convert month name to month number
        month_number = datetime.strptime(month.strip(), "%B").month

        # Using the current year
        year = datetime.now().year - 1
        print(year)

        start_date = f"{year}-{month_number:02d}-01"

        last_day = calendar.monthrange(year, month_number)[1]
        end_date = f"{year}-{month_number:02d}-{last_day:02d}"

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_min,temperature_2m_max,weather_code",
            "timezone": "auto",
        }

        response = requests.get(
            url,
            params=params,
            timeout=6
        )

        response.raise_for_status()

        daily = response.json().get("daily", {})

        min_temps = [
            temp for temp in daily.get("temperature_2m_min", [])
            if temp is not None
        ]

        max_temps = [
            temp for temp in daily.get("temperature_2m_max", [])
            if temp is not None
        ]

        if not min_temps or not max_temps:
            return {
                "status": "error",
                "message": "No weather data available for this month."
            }

        return {
            "status": "success",
            "month": month,
            "temp_range_c": [
                min(min_temps),
                max(max_temps)
            ]
        }

    except ValueError:
        return {
            "status": "error",
            "message": (
                "Invalid month. Please provide a full month name, "
                "for example 'December'."
            )
        }

    except requests.Timeout:
        return {
            "status": "error",
            "message": "Weather API request timed out."
        }

    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Weather API request failed: {str(e)}"
        }

if __name__ == "__main__":
    # result = get_weather_forecast.invoke({
    #     "latitude": 35.6762,
    #     "longitude": 139.6503,
    #     "month": "October"
    # })

    # print(result)

    print(
    search_destination_guide.invoke(
        "Tokyo packing tips"
    )
)
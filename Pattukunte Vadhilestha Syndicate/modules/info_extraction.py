from langchain_groq import ChatGroq

def initialize_groq():
    """Initialize ChatGroq."""
    return ChatGroq(
        temperature=0,
        groq_api_key='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',
        model_name="llama-3.3-70b-versatile"
    )

def extract_info(llm, detected_info_str):
    """Extract structured information from detected text."""
    query = """
    The following text contains details from an identity card or similar official document. Your task is to extract the relevant structured information as requested below. Ensure you extract all available fields, leaving blanks where information is missing. Pay attention to the format and ensure it matches what would be seen on an official identity card.

    Please extract the following details:
    - Name (Full name as it appears on the ID)
    - ID Number (The unique identification number associated with the individual. This could be a passport number, driver's license number, or another identifier. The ID number will be numeric or alphanumeric and should be carefully selected based on context and pattern in the document. Pay special attention to ensure the ID number is correctly identified and matches the expected format of an identity card).
    - Date of Birth (DOB in the format 'DD/MM/YYYY')
    - Address (The complete residential address, including street, city, postal code, and country if available)
    - Other Information (Any additional details like nationality, gender, etc.)

    The output should be in the following dictionary format:
    {
        "Card Type": "Identity Card",  # Specify the type of document (for example, "Identity Card", "Driving License", etc.)
        "Name": "",  # Full name as it appears on the document
        "ID Number": "",  # The unique identification number. Ensure this is correctly identified.
        "Date of Birth": "",  # The birthdate in 'DD/MM/YYYY' format
        "Address": "",  # Full address
        "Other Information": ""  # Any other relevant details that can be extracted
    }

    Important Guidelines:
    1. Double-check that all extracted details are correct, and format them precisely as they would appear on the official ID card.
    2. If the text is unclear or there are multiple possible interpretations, include a note or leave it blank where necessary.
    3. Ensure that the date is always formatted correctly as 'DD/MM/YYYY'.
    4. If the text mentions a specific card type (e.g., passport, driver’s license, etc.), mention it in the 'Card Type' field.
    5. Pay particular attention to the **ID Number**: There might be multiple numbers in the text, but the correct one should match the format of the identity card (for example, a sequence of numbers, alphanumeric code, etc.). Ensure to identify the right ID number contextually.
    6. If certain fields cannot be confidently extracted, leave them blank but ensure the overall structure remains intact.

    Here is the text:
    """ + detected_info_str

    response = llm.invoke(query)
    return response.content

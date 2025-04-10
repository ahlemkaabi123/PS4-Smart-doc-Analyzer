document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.getElementById("uploadForm");
  const qaForm = document.getElementById("qaForm");
  const documentSelect = document.getElementById("documentSelect");
  const uploadStatus = document.getElementById("uploadStatus");
  const answerDiv = document.getElementById("answer");

  // Handle document upload
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("document");
    const file = fileInput.files[0];

    if (!file) {
      showStatus("Please select a file", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      showStatus("Uploading and processing document...", "info");
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Error uploading document");
      }

      const data = await response.json();
      showStatus(data.message, "success");
      updateDocumentSelect();
      fileInput.value = ""; // Clear the file input
    } catch (error) {
      console.error("Upload error:", error);
      showStatus(error.message || "Error uploading document", "error");
    }
  });

  // Handle question submission
  qaForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const documentName = documentSelect.value;
    const question = document.getElementById("question").value;

    if (!documentName || !question) {
      showAnswer("Please select a document and enter a question", "error");
      return;
    }

    try {
      showAnswer("Searching for answers...", "info");

      // Create FormData object
      const formData = new FormData();
      formData.append("document_name", documentName);
      formData.append("question", question);

      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Error getting answer");
      }

      const data = await response.json();
      displayAnswer(data);
    } catch (error) {
      console.error("Question error:", error);
      showAnswer(error.message || "Error getting answer", "error");
    }
  });

  // Update document select dropdown
  async function updateDocumentSelect() {
    try {
      const response = await fetch("http://localhost:8000/documents");
      if (!response.ok) {
        throw new Error("Failed to fetch documents");
      }
      const documents = await response.json();

      documentSelect.innerHTML = '<option value="">Select a document</option>';
      documents.forEach((doc) => {
        const option = document.createElement("option");
        option.value = doc;
        option.textContent = doc;
        documentSelect.appendChild(option);
      });
    } catch (error) {
      console.error("Error fetching documents:", error);
      showStatus("Error loading document list", "error");
    }
  }

  // Helper functions for displaying status and answers
  function showStatus(message, type = "info") {
    uploadStatus.textContent = message;
    uploadStatus.className = type;
  }

  function showAnswer(message, type = "info") {
    answerDiv.innerHTML = `<p class="${type}">${message}</p>`;
  }

  function displayAnswer(data) {
    let html = '<div class="answer-content">';

    // Display the answer
    if (data.answer) {
      html += `<div class="answer-text">${data.answer}</div>`;
    }

    // Display sources if available
    if (data.sources && data.sources.length > 0) {
      html += '<div class="sources">';
      html += "<h3>Sources:</h3>";
      data.sources.forEach((source, index) => {
        html += `<div class="source">
                    <p class="source-page"><strong>Page ${
                      source.page + 1
                    }:</strong></p>
                    <p class="source-content">${source.content}</p>
                </div>`;
      });
      html += "</div>";
    }

    html += "</div>";
    answerDiv.innerHTML = html;
  }

  // Initial document list update
  updateDocumentSelect();
});

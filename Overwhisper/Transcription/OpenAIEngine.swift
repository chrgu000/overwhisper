import Foundation

struct OpenAIEngine: TranscriptionEngine {
    static let defaultBaseURL = "https://api.openai.com/v1"
    static let defaultModel = "whisper-1"

    private let apiKey: String
    private let baseURL: String
    private let model: String
    private let language: String
    private let translateToEnglish: Bool
    private let customVocabulary: String
    private static let requestTimeoutSeconds: TimeInterval = 90

    /// Accepts either an API root (".../v1") or a full ".../audio/transcriptions" URL.
    private var endpointURL: URL? {
        var root = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if root.isEmpty { root = Self.defaultBaseURL }
        while root.hasSuffix("/") { root.removeLast() }
        for suffix in ["/audio/transcriptions", "/audio/translations"] where root.hasSuffix(suffix) {
            root.removeLast(suffix.count)
        }
        return URL(string: root + (translateToEnglish ? "/audio/translations" : "/audio/transcriptions"))
    }

    init(
        apiKey: String,
        baseURL: String = OpenAIEngine.defaultBaseURL,
        model: String = OpenAIEngine.defaultModel,
        language: String = "auto",
        translateToEnglish: Bool = false,
        customVocabulary: String = ""
    ) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        let trimmedModel = model.trimmingCharacters(in: .whitespacesAndNewlines)
        self.model = trimmedModel.isEmpty ? Self.defaultModel : trimmedModel
        self.language = language
        self.translateToEnglish = translateToEnglish
        self.customVocabulary = customVocabulary
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard !apiKey.isEmpty else {
            throw OpenAIError.missingAPIKey
        }

        // Read audio file data
        let audioData = try Data(contentsOf: audioURL)

        // Create multipart form data
        let boundary = UUID().uuidString
        guard let endpointURL else {
            throw OpenAIError.invalidBaseURL
        }
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = Self.requestTimeoutSeconds
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // Add file field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(audioData)
        body.append("\r\n".data(using: .utf8)!)

        // Add model field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"model\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(model)\r\n".data(using: .utf8)!)

        // Add language hint (transcriptions only; translations always output English)
        if !translateToEnglish && language != "auto" && !language.isEmpty {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"language\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(language)\r\n".data(using: .utf8)!)
        }

        // Add response format field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"response_format\"\r\n\r\n".data(using: .utf8)!)
        body.append("json\r\n".data(using: .utf8)!)

        // Add prompt field for custom vocabulary
        if !customVocabulary.isEmpty {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"prompt\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(customVocabulary)\r\n".data(using: .utf8)!)
        }

        // Close boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        // Make request
        let (data, response) = try await URLSession.shared.data(for: request)

        // Check response
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAIError.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            // Try to parse error message
            if let errorResponse = try? JSONDecoder().decode(OpenAIErrorResponse.self, from: data) {
                throw OpenAIError.apiError(errorResponse.error.message)
            }
            throw OpenAIError.httpError(httpResponse.statusCode)
        }

        // Parse response
        let transcriptionResponse = try JSONDecoder().decode(OpenAITranscriptionResponse.self, from: data)

        return transcriptionResponse.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct OpenAITranscriptionResponse: Codable {
    let text: String
}

struct OpenAIErrorResponse: Codable {
    let error: OpenAIErrorDetail
}

struct OpenAIErrorDetail: Codable {
    let message: String
    let type: String?
    let code: String?
}

enum OpenAIError: LocalizedError {
    case missingAPIKey
    case invalidBaseURL
    case invalidResponse
    case httpError(Int)
    case apiError(String)
    case decodingError

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenAI API key is not configured"
        case .invalidBaseURL:
            return "OpenAI-compatible base URL is invalid"
        case .invalidResponse:
            return "Invalid response from OpenAI API"
        case .httpError(let code):
            return "HTTP error: \(code)"
        case .apiError(let message):
            return message
        case .decodingError:
            return "Failed to decode API response"
        }
    }
}

class AppConfig {
  // 🌐 UPDATE THIS WITH YOUR ACTUAL CLOUDFLARE TUNNEL URL
  static const String CLOUDFLARE_TUNNEL_URL = 'https://columbus-indicating-intensive-regardless.trycloudflare.com';
  
  // API Configuration
  static String get apiBaseUrl => CLOUDFLARE_TUNNEL_URL;
  static const Duration apiTimeout = Duration(seconds: 30);
  
  // Hardcoded supported languages (no configuration screen needed)
  static const Map<String, String> supportedLanguages = {
    'en': 'English',
    'ur': 'Urdu',
    'hi': 'Hindi',
    'ar': 'Arabic',
  };
  
  // Hardcoded video formats
  static const List<String> supportedVideoFormats = [
    'mp4', 'avi', 'mov', 'mkv', 'wmv', '3gp', 'webm', 'm4v'
  ];
  
  // File size limit
  static const int maxFileSizeMb = 500;
  
  // App metadata
  static const String appName = 'AutoDub';
  static const String appVersion = '1.0.0';
  static const String userAgent = 'AutoDub-Flutter/1.0';
}

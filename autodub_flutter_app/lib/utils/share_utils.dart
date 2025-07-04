import 'package:share_plus/share_plus.dart';

class ShareUtils {
  static Future<void> shareApp() async {
    try {
      await SharePlus.instance.share(
        ShareParams(
          text: 'Check out AutoDub - AI-Powered Video Dubbing App!\n\n'
              'Transform your videos with automatic language dubbing using advanced AI technology. '
              'Support for multiple languages including English, Urdu, Hindi, and Arabic.\n\n'
              '🔹 Auto speaker detection & gender-aware dubbing\n'
              '🔹 Accurate speech recognition and translation\n'
              '🔹 Natural-sounding Text-to-Speech (TTS)\n'
              '🔹 Lip-sync integration\n\n'
              'Download now and break language barriers effortlessly!\n\n'
              'Developer: Muhammad Danyal\n'
              'Contact: +92 317 9463062\n'
              'Email: muhammaddanyal0016@gmail.com',
          subject: 'AutoDub - AI-Powered Video Dubbing App',
        ),
      );
    } catch (e) {
      // Handle error silently
    }
  }

  static Future<void> shareText(String text, {String? subject}) async {
    try {
      await SharePlus.instance.share(
        ShareParams(
          text: text,
          subject: subject,
        ),
      );
    } catch (e) {
      // Handle error silently
    }
  }
} 
import 'package:autodub_flutter_app/screens/splash_screen.dart';
import 'package:autodub_flutter_app/theme/app_theme.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(AutoDubApp());
}

class AutoDubApp extends StatelessWidget {
  const AutoDubApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'AutoDub',
      theme: AppTheme.darkTheme,
      home: SplashScreen(),
    );
  }
}

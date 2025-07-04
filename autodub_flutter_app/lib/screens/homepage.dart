import 'package:autodub_flutter_app/screens/uploadvideopage.dart';
import 'package:autodub_flutter_app/screens/about_screen.dart';
import 'package:autodub_flutter_app/screens/help_support_screen.dart';
import 'package:autodub_flutter_app/utils/share_utils.dart';
import 'package:flutter/material.dart';

import 'package:page_transition/page_transition.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:flutter_animate/flutter_animate.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: _buildDrawer(context),
      body: Stack(
        children: [
          _buildBackground(),
          SafeArea(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildHeader(context),
                  _buildHeroSection(context),
                  _buildFeaturesSection(),
                  _buildHowItWorksSection(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBackground() {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF293431), Color(0xFF45AA96)],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            'AutoDub',
            style: GoogleFonts.poppins(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          Builder(
            builder: (BuildContext innerContext) {
              return IconButton(
                icon: Icon(Icons.menu, color: Colors.white),
                onPressed: () {
                  Scaffold.of(innerContext).openDrawer();
                },
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildHeroSection(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 24, vertical: 40),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          AnimatedTextKit(
            animatedTexts: [
              TypewriterAnimatedText(
                'Effortless Video Dubbing',
                textStyle: GoogleFonts.poppins(
                  fontSize: 36,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
                speed: const Duration(milliseconds: 100),
              ),
            ],
            totalRepeatCount: 1,
            displayFullTextOnTap: true,
            stopPauseOnTap: true,
          ),
          SizedBox(height: 16),
          Text(
            'Transform your videos into multiple languages with AI-powered dubbing',
            style: GoogleFonts.poppins(
              fontSize: 16,
              color: Colors.white70,
            ),
          )
              .animate()
              .fadeIn(duration: 800.ms, delay: 500.ms)
              .slideX(begin: -0.2, end: 0),
          SizedBox(height: 32),
          _buildAnimatedButton(context),
        ],
      ),
    );
  }

  Widget _buildAnimatedButton(BuildContext context) {
    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          PageTransition(
            type: PageTransitionType.rightToLeft,
            child: UploadVideoPage(),
          ),
        );
      },
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        decoration: BoxDecoration(
          color: Color(0xFF05CEA8),
          borderRadius: BorderRadius.circular(30),
          boxShadow: [
            BoxShadow(
              color: Color(0xFF05CEA8).withOpacity(0.5),
              blurRadius: 10,
              offset: Offset(0, 5),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Get Started',
              style: GoogleFonts.poppins(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
            SizedBox(width: 8),
            Icon(
              Icons.arrow_forward,
              color: Colors.white,
            ),
          ],
        ),
      ),
    )
        .animate(onPlay: (controller) => controller.repeat())
        .shimmer(duration: 1500.ms, color: Colors.white.withOpacity(0.5))
        .animate()
        .fadeIn(duration: 800.ms, delay: 800.ms)
        .slideY(begin: 0.2, end: 0);
  }

  Widget _buildFeaturesSection() {
    final features = [
      {
        'icon': Icons.language,
        'title': 'Multiple Languages',
        'description': 'Support for various languages'
      },
      {
        'icon': Icons.psychology,
        'title': 'AI-Powered',
        'description': 'Advanced AI for natural-sounding dubs'
      },
      {
        'icon': Icons.speed,
        'title': 'Fast Processing',
        'description': 'Quick turnaround for your videos'
      },
      {
        'icon': Icons.high_quality,
        'title': 'High-Quality Output',
        'description': 'Crystal clear audio and perfect lip-sync'
      },
      {
        'icon': Icons.security,
        'title': 'Secure & Private',
        'description': 'Your content is always protected'
      }
    ];

    return Container(
      padding: EdgeInsets.symmetric(vertical: 40),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Text(
              'Features',
              style: GoogleFonts.poppins(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ).animate().fadeIn(duration: 600.ms).slideX(begin: -0.2, end: 0),
          ),
          SizedBox(height: 24),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: features.asMap().entries.map((entry) {
                int idx = entry.key;
                Map<String, dynamic> feature = entry.value;
                return _buildFeatureCard(feature, idx);
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureCard(Map<String, dynamic> feature, int index) {
    return Container(
      width: 200,
      height: 200,
      margin: EdgeInsets.only(left: 24, right: 8, bottom: 16),
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            feature['icon'] as IconData,
            size: 36,
            color: Color(0xFF05CEA8),
          )
              .animate(onPlay: (controller) => controller.repeat())
              .scale(
                begin: Offset(1, 1),
                end: Offset(1.2, 1.2),
                duration: 1000.ms,
              )
              .then(delay: 500.ms)
              .scale(
                begin: Offset(1.2, 1.2),
                end: Offset(1, 1),
                duration: 1000.ms,
              ),
          SizedBox(height: 12),
          Text(
            feature['title'] as String,
            style: GoogleFonts.poppins(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          SizedBox(height: 8),
          Expanded(
            child: Text(
              feature['description'] as String,
              style: GoogleFonts.poppins(
                fontSize: 13,
                color: Colors.white70,
              ),
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    )
        .animate(delay: (300 * index).ms)
        .fadeIn(duration: 500.ms)
        .slideY(begin: 0.2, end: 0)
        .then(delay: 200.ms)
        .shimmer(duration: 1200.ms, color: Colors.white.withOpacity(0.2));
  }

  Widget _buildHowItWorksSection() {
    final steps = [
      {'number': '1', 'title': 'Upload', 'description': 'Upload your video'},
      {'number': '2', 'title': 'Select', 'description': 'Choose languages'},
      {'number': '3', 'title': 'Process', 'description': 'AI dubs your video'},
      {
        'number': '4',
        'title': 'Download',
        'description': 'Get your dubbed video'
      },
    ];

    return Container(
      padding: EdgeInsets.symmetric(vertical: 40, horizontal: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'How It Works',
            style: GoogleFonts.poppins(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          SizedBox(height: 24),
          ...steps.map((step) => _buildStepItem(step)),
        ],
      ),
    );
  }

  Widget _buildStepItem(Map<String, String> step) {
    return Container(
      margin: EdgeInsets.only(bottom: 24),
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(15),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: Offset(0, 5),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: Color(0xFF05CEA8),
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                step['number']!,
                style: GoogleFonts.poppins(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
          ),
          SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  step['title']!,
                  style: GoogleFonts.poppins(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  step['description']!,
                  style: GoogleFonts.poppins(
                    fontSize: 14,
                    color: Colors.white70,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDrawer(BuildContext context) {
    return Drawer(
      child: Container(
        color: Color(0xFF293431),
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [Color(0xFF45AA96), Color(0xFF05CEA8)],
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Icon(
                    Icons.record_voice_over,
                    size: 50,
                    color: Colors.white,
                  )
                      .animate(onPlay: (controller) => controller.repeat())
                      .scale(
                        begin: Offset(1, 1),
                        end: Offset(1.2, 1.2),
                        duration: 1000.ms,
                      )
                      .then(delay: 500.ms)
                      .scale(
                        begin: Offset(1.2, 1.2),
                        end: Offset(1, 1),
                        duration: 1000.ms,
                      ),
                  SizedBox(height: 10),
                  Text(
                    'AutoDub',
                    style: GoogleFonts.poppins(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'AI-Powered Video Dubbing',
                    style: GoogleFonts.poppins(
                      color: Colors.white.withOpacity(0.8),
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),
            _buildDrawerItem(
              icon: Icons.home,
              title: 'Home',
              onTap: () {
                Navigator.pop(context);
              },
            ),
            Divider(color: Colors.white24),
            _buildDrawerItem(
              icon: Icons.help,
              title: 'Help & Support',
              onTap: () {
                Navigator.pop(context);
                Navigator.push(
                  context,
                  PageTransition(
                    type: PageTransitionType.rightToLeft,
                    child: HelpSupportScreen(),
                  ),
                );
              },
            ),
            _buildDrawerItem(
              icon: Icons.info,
              title: 'About',
              onTap: () {
                Navigator.pop(context);
                Navigator.push(
                  context,
                  PageTransition(
                    type: PageTransitionType.rightToLeft,
                    child: AboutScreen(),
                  ),
                );
              },
            ),
            _buildDrawerItem(
              icon: Icons.share,
              title: 'Share App',
              onTap: () {
                Navigator.pop(context);
                ShareUtils.shareApp();
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDrawerItem({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return ListTile(
      leading: Icon(
        icon,
        color: Color(0xFF05CEA8),
        size: 24,
      ),
      title: Text(
        title,
        style: GoogleFonts.poppins(
          color: Colors.white,
          fontSize: 16,
        ),
      ),
      onTap: onTap,
      hoverColor: Colors.white.withOpacity(0.1),
      selectedTileColor: Colors.white.withOpacity(0.1),
    );
  }
}

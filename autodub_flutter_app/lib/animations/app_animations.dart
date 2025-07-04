import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

class FadeSlideTransition extends StatelessWidget {
  final Widget child;
  final Animation<double> animation;

  const FadeSlideTransition(
      {super.key, required this.child, required this.animation});

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: animation,
      builder: (context, child) {
        return Opacity(
          opacity: animation.value,
          child: Transform.translate(
            offset: Offset(0, 50 * (1 - animation.value)),
            child: child,
          ),
        );
      },
      child: child,
    );
  }
}

class ScaleAnimation extends StatelessWidget {
  final Widget child;
  final Animation<double> animation;

  const ScaleAnimation(
      {super.key, required this.child, required this.animation});

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: animation,
      builder: (context, child) {
        return Transform.scale(
          scale: 1.0 + (animation.value * 0.1),
          child: child,
        );
      },
      child: child,
    );
  }
}

class FadeSlideAnimation extends StatelessWidget {
  final Widget child;
  final double begin;
  final double end;
  final int delay;
  final int duration;

  const FadeSlideAnimation({
    super.key,
    required this.child,
    this.begin = -0.2,
    this.end = 0,
    this.delay = 0,
    this.duration = 800,
  });

  @override
  Widget build(BuildContext context) {
    return child
        .animate()
        .fadeIn(duration: duration.ms, delay: delay.ms)
        .slideX(begin: begin, end: end);
  }
}

class ShimmerAnimation extends StatelessWidget {
  final Widget child;
  final int duration;

  const ShimmerAnimation({
    super.key,
    required this.child,
    this.duration = 1500,
  });

  @override
  Widget build(BuildContext context) {
    return child
        .animate(onPlay: (controller) => controller.repeat())
        .shimmer(duration: duration.ms, color: Colors.white.withOpacity(0.5));
  }
}

extension IntExtension on int {
  Duration get ms => Duration(milliseconds: this);
}

import 'package:flutter/material.dart';

class OptimizedScaleAnimation extends StatefulWidget {
  final Widget child;
  final bool enabled;
  final Duration duration;
  final double scaleRange;

  const OptimizedScaleAnimation({
    Key? key,
    required this.child,
    this.enabled = true,
    this.duration = const Duration(seconds: 3),
    this.scaleRange = 0.1,
  }) : super(key: key);

  @override
  _OptimizedScaleAnimationState createState() => _OptimizedScaleAnimationState();
}

class _OptimizedScaleAnimationState extends State<OptimizedScaleAnimation>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.duration,
      vsync: this,
    );
    _animation = Tween<double>(
      begin: 1.0,
      end: 1.0 + widget.scaleRange,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));

    if (widget.enabled) {
      _controller.repeat(reverse: true);
    }
  }

  @override
  void didUpdateWidget(OptimizedScaleAnimation oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.enabled != oldWidget.enabled) {
      if (widget.enabled) {
        _controller.repeat(reverse: true);
      } else {
        _controller.stop();
        _controller.reset();
      }
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.enabled) {
      return widget.child;
    }

    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Transform.scale(
          scale: _animation.value,
          child: widget.child,
        );
      },
    );
  }
}

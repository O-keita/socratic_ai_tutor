class Course {
  final String id;
  final String title;
  final String description;
  final String thumbnail;
  final List<Module> modules;
  final int totalLessons;
  final String difficulty;
  final String duration;

  Course({
    required this.id,
    required this.title,
    required this.description,
    required this.thumbnail,
    required this.modules,
    required this.totalLessons,
    required this.difficulty,
    required this.duration,
  });

  factory Course.fromJson(Map<String, dynamic> json) {
    return Course(
      id: json['id'],
      title: json['title'],
      description: json['description'] ?? '',
      thumbnail: json['thumbnail'] ?? json['imageUrl'] ?? '',
      totalLessons: json['totalLessons'] ?? 0,
      difficulty: json['difficulty'] ?? 'Beginner',
      duration: json['duration'] ?? '',
      modules: json['modules'] != null
          ? (json['modules'] as List).map((m) => Module.fromJson(m)).toList()
          : [],
    );
  }

  int get completedLessons {
    int count = 0;
    for (var module in modules) {
      for (var chapter in module.chapters) {
        for (var lesson in chapter.lessons) {
          if (lesson.isCompleted) count++;
        }
      }
    }
    return count;
  }

  double get progress {
    if (totalLessons == 0) return 0;
    return completedLessons / totalLessons;
  }
}

class Module {
  final String id;
  final String title;
  final String description;
  final List<Chapter> chapters;
  final int orderIndex;

  Module({
    required this.id,
    required this.title,
    required this.description,
    required this.chapters,
    required this.orderIndex,
  });

  factory Module.fromJson(Map<String, dynamic> json) {
    return Module(
      id: json['id'],
      title: json['title'],
      description: json['description'] ?? '',
      orderIndex: json['orderIndex'] ?? 0,
      chapters: json['chapters'] != null
          ? (json['chapters'] as List).map((c) => Chapter.fromJson(c)).toList()
          : [],
    );
  }

  bool get isCompleted {
    return chapters.every((chapter) => chapter.isCompleted);
  }

  int get completedChapters {
    return chapters.where((chapter) => chapter.isCompleted).length;
  }
}

class Chapter {
  final String id;
  final String title;
  final List<Lesson> lessons;
  final int orderIndex;

  Chapter({
    required this.id,
    required this.title,
    required this.lessons,
    required this.orderIndex,
  });

  factory Chapter.fromJson(Map<String, dynamic> json) {
    return Chapter(
      id: json['id'],
      title: json['title'],
      orderIndex: json['orderIndex'] ?? 0,
      lessons: json['lessons'] != null
          ? (json['lessons'] as List).map((l) => Lesson.fromJson(l)).toList()
          : [],
    );
  }

  bool get isCompleted {
    return lessons.every((lesson) => lesson.isCompleted);
  }

  int get completedLessons {
    return lessons.where((lesson) => lesson.isCompleted).length;
  }
}

class Lesson {
  final String id;
  final String title;
  String content;
  final String? contentFile;
  final List<String> images;
  final int orderIndex;
  bool isCompleted;
  final String? keyPoints;
  final List<String>? reflectionQuestions;

  Lesson({
    required this.id,
    required this.title,
    this.content = '',
    this.contentFile,
    this.images = const [],
    required this.orderIndex,
    this.isCompleted = false,
    this.keyPoints,
    this.reflectionQuestions,
  });

  factory Lesson.fromJson(Map<String, dynamic> json) {
    return Lesson(
      id: json['id'],
      title: json['title'],
      content: json['content'] ?? '',
      contentFile: json['contentFile'] ?? json['contentPath'],
      orderIndex: json['orderIndex'] ?? 0,
      images: json['images'] != null
          ? List<String>.from(json['images'])
          : [],
      keyPoints: json['keyPoints'],
      reflectionQuestions: json['reflectionQuestions'] != null
          ? List<String>.from(json['reflectionQuestions'])
          : null,
    );
  }
}


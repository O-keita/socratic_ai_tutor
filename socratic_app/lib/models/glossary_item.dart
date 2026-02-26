class GlossaryItem {
  final String term;
  final String definition;
  final String category;

  GlossaryItem({
    required this.term,
    required this.definition,
    required this.category,
  });

  factory GlossaryItem.fromJson(Map<String, dynamic> json) {
    return GlossaryItem(
      term: json['term'],
      definition: json['definition'],
      category: json['category'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'term': term,
      'definition': definition,
      'category': category,
    };
  }
}

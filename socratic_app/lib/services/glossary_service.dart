import 'dart:convert';
import 'package:flutter/services.dart';
import '../models/glossary_item.dart';

class GlossaryService {
  static final GlossaryService _instance = GlossaryService._internal();
  factory GlossaryService() => _instance;
  GlossaryService._internal();

  List<GlossaryItem> _items = [];
  bool _isLoading = false;

  List<GlossaryItem> get items => _items;
  bool get isLoading => _isLoading;

  Future<void> loadGlossary() async {
    if (_items.isNotEmpty) return;
    
    _isLoading = true;
    try {
      final String jsonString = await rootBundle.loadString('assets/glossary/terms.json');
      final List<dynamic> jsonData = jsonDecode(jsonString);
      _items = jsonData.map((item) => GlossaryItem.fromJson(item)).toList();
      
      // Sort items alphabetically
      _items.sort((a, b) => a.term.toLowerCase().compareTo(b.term.toLowerCase()));
      
      print('GlossaryService: Loaded ${_items.length} terms');
    } catch (e) {
      print('Error loading glossary data: $e');
    } finally {
      _isLoading = false;
    }
  }

  List<String> getCategories() {
    return _items.map((item) => item.category).toSet().toList()..sort();
  }

  List<GlossaryItem> getItemsByCategory(String category) {
    return _items.where((item) => item.category == category).toList();
  }
  
  List<GlossaryItem> searchTerms(String query) {
    if (query.isEmpty) return _items;
    final lowercaseQuery = query.toLowerCase();
    return _items.where((item) => 
      item.term.toLowerCase().contains(lowercaseQuery) || 
      item.definition.toLowerCase().contains(lowercaseQuery)
    ).toList();
  }
}

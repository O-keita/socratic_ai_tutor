import 'package:flutter/material.dart';
import '../models/glossary_item.dart';
import '../services/glossary_service.dart';
import '../theme/app_theme.dart';
import 'package:provider/provider.dart';
import '../services/theme_service.dart';

class GlossaryScreen extends StatefulWidget {
  const GlossaryScreen({super.key});

  @override
  State<GlossaryScreen> createState() => _GlossaryScreenState();
}

class _GlossaryScreenState extends State<GlossaryScreen> {
  final GlossaryService _glossaryService = GlossaryService();
  String _searchQuery = '';
  String _selectedCategory = 'All';
  List<GlossaryItem> _filteredItems = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    await _glossaryService.loadGlossary();
    if (mounted) {
      setState(() {
        _filteredItems = _glossaryService.items;
        _isLoading = false;
      });
    }
  }

  void _filterTerms() {
    List<GlossaryItem> results = _glossaryService.searchTerms(_searchQuery);
    if (_selectedCategory != 'All') {
      results = results.where((item) => item.category == _selectedCategory).toList();
    }
    setState(() {
      _filteredItems = results;
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Key Terms Glossary'),
        elevation: 0,
        backgroundColor: Colors.transparent,
      ),
      extendBodyBehindAppBar: true,
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildSearchAndFilter(isDark),
              Expanded(
                child: _isLoading
                    ? const Center(child: CircularProgressIndicator(color: AppTheme.accentOrange))
                    : _buildTermList(isDark),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSearchAndFilter(bool isDark) {
    final categories = ['All', ..._glossaryService.getCategories()];
    
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          TextField(
            onChanged: (value) {
              _searchQuery = value;
              _filterTerms();
            },
            style: TextStyle(color: isDark ? Colors.white : Colors.black87),
            decoration: InputDecoration(
              hintText: 'Search terms...',
              hintStyle: TextStyle(color: isDark ? Colors.white54 : Colors.black45),
              prefixIcon: Icon(Icons.search, color: isDark ? AppTheme.accentOrange : AppTheme.accentOrange),
              filled: true,
              fillColor: isDark ? Colors.white.withValues(alpha: 0.05) : Colors.black.withValues(alpha: 0.05),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide.none,
              ),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 40,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: categories.length,
              itemBuilder: (context, index) {
                final category = categories[index];
                final isSelected = _selectedCategory == category;
                return Padding(
                  padding: const EdgeInsets.only(right: 8.0),
                  child: FilterChip(
                    label: Text(category),
                    selected: isSelected,
                    onSelected: (selected) {
                      setState(() {
                        _selectedCategory = category;
                        _filterTerms();
                      });
                    },
                    selectedColor: AppTheme.accentOrange.withValues(alpha: 0.2),
                    checkmarkColor: AppTheme.accentOrange,
                    labelStyle: TextStyle(
                      color: isSelected 
                          ? AppTheme.accentOrange 
                          : (isDark ? Colors.white70 : Colors.black87),
                      fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                    ),
                    backgroundColor: isDark ? Colors.white.withValues(alpha: 0.05) : Colors.black.withValues(alpha: 0.05),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                      side: BorderSide(
                        color: isSelected ? AppTheme.accentOrange : Colors.transparent,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTermList(bool isDark) {
    if (_filteredItems.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.search_off, size: 64, color: isDark ? Colors.white24 : Colors.black26),
            const SizedBox(height: 16),
            Text(
              'No terms found',
              style: TextStyle(
                color: isDark ? Colors.white60 : Colors.black54,
                fontSize: 18,
              ),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      itemCount: _filteredItems.length,
      itemBuilder: (context, index) {
        final item = _filteredItems[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          color: isDark ? Colors.white.withValues(alpha: 0.05) : Colors.white,
          elevation: isDark ? 0 : 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: isDark 
                ? BorderSide(color: Colors.white.withValues(alpha: 0.1)) 
                : BorderSide.none,
          ),
          child: ExpansionTile(
            title: Text(
              item.term,
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
                color: isDark ? Colors.white : Colors.black87,
              ),
            ),
            subtitle: Text(
              item.category,
              style: TextStyle(
                color: AppTheme.accentOrange,
                fontSize: 12,
              ),
            ),
            iconColor: AppTheme.accentOrange,
            collapsedIconColor: isDark ? Colors.white54 : Colors.black45,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Text(
                  item.definition,
                  style: TextStyle(
                    fontSize: 15,
                    color: isDark ? Colors.white.withValues(alpha: 0.8) : Colors.black87,
                    height: 1.5,
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

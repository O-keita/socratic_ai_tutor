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
    if (mounted) setState(() { _filteredItems = _glossaryService.items; _isLoading = false; });
  }

  void _filterTerms() {
    List<GlossaryItem> results = _glossaryService.searchTerms(_searchQuery);
    if (_selectedCategory != 'All') {
      results = results.where((item) => item.category == _selectedCategory).toList();
    }
    setState(() { _filteredItems = results; });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    final bgColor = isDark ? AppTheme.primaryDark : const Color(0xFFFEF6EE);

    return Scaffold(
      body: Container(
        color: bgColor,
        child: SafeArea(
          child: Column(
            children: [
              // ── Header ──────────────────────────
              Container(
                padding: const EdgeInsets.fromLTRB(8, 8, 20, 16),
                decoration: BoxDecoration(
                  gradient: isDark ? AppTheme.headerGradientDark : AppTheme.headerGradientLight,
                  borderRadius: const BorderRadius.only(bottomLeft: Radius.circular(32), bottomRight: Radius.circular(32)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        IconButton(
                          icon: Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(color: isDark ? AppTheme.surfaceCard : Colors.white, shape: BoxShape.circle),
                            child: Icon(Icons.arrow_back_ios_new, size: 16, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
                          ),
                          onPressed: () => Navigator.pop(context),
                        ),
                        const SizedBox(width: 4),
                        Text('Key Terms', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                      ],
                    ),
                    const SizedBox(height: 12),
                    // Search bar
                    Padding(
                      padding: const EdgeInsets.only(left: 12),
                      child: TextField(
                        onChanged: (v) { _searchQuery = v; _filterTerms(); },
                        style: TextStyle(color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary, fontSize: 14),
                        decoration: InputDecoration(
                          hintText: 'Search terms...',
                          hintStyle: TextStyle(color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted),
                          prefixIcon: const Icon(Icons.search, color: AppTheme.accentOrange, size: 20),
                          filled: true,
                          fillColor: isDark ? AppTheme.surfaceCard : Colors.white,
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
                          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // ── Category chips ──────────────────
              SizedBox(
                height: 52,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.fromLTRB(20, 12, 20, 4),
                  itemCount: ['All', ..._glossaryService.getCategories()].length,
                  itemBuilder: (context, index) {
                    final categories = ['All', ..._glossaryService.getCategories()];
                    final cat = categories[index];
                    final isSel = _selectedCategory == cat;
                    return Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: GestureDetector(
                        onTap: () => setState(() { _selectedCategory = cat; _filterTerms(); }),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                          decoration: BoxDecoration(
                            color: isSel ? AppTheme.accentOrange : (isDark ? AppTheme.surfaceCard : Colors.white),
                            borderRadius: BorderRadius.circular(20),
                            border: isSel ? null : Border.all(color: isDark ? AppTheme.primaryLight.withValues(alpha: 0.2) : Colors.grey.withValues(alpha: 0.15)),
                          ),
                          child: Text(cat, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: isSel ? Colors.white : (isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary))),
                        ),
                      ),
                    );
                  },
                ),
              ),

              // ── Term list ───────────────────────
              Expanded(
                child: _isLoading
                    ? const Center(child: CircularProgressIndicator(color: AppTheme.accentOrange))
                    : _filteredItems.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.search_off, size: 56, color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted),
                                const SizedBox(height: 12),
                                Text('No terms found', style: TextStyle(color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary, fontSize: 16)),
                              ],
                            ),
                          )
                        : ListView.builder(
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                            itemCount: _filteredItems.length,
                            itemBuilder: (context, index) {
                              final item = _filteredItems[index];
                              return Container(
                                margin: const EdgeInsets.only(bottom: 10),
                                decoration: BoxDecoration(
                                  color: isDark ? AppTheme.surfaceCard : Colors.white,
                                  borderRadius: BorderRadius.circular(18),
                                  boxShadow: isDark ? null : [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 8, offset: const Offset(0, 2))],
                                ),
                                child: ExpansionTile(
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                                  collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                                  title: Text(item.term, style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                                  subtitle: Text(item.category, style: const TextStyle(color: AppTheme.accentOrange, fontSize: 12, fontWeight: FontWeight.w500)),
                                  iconColor: AppTheme.accentOrange,
                                  collapsedIconColor: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted,
                                  children: [
                                    Padding(
                                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                                      child: Text(item.definition, style: TextStyle(fontSize: 14, color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary, height: 1.5)),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

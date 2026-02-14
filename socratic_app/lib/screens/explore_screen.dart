import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../widgets/gradient_card.dart';

class ExploreScreen extends StatefulWidget {
  const ExploreScreen({super.key});

  @override
  State<ExploreScreen> createState() => _ExploreScreenState();
}

class _ExploreScreenState extends State<ExploreScreen> {
  final TextEditingController _searchController = TextEditingController();
  int _selectedTab = 0;

  final List<String> _tabs = ['Favorites', 'Research', 'Self-Study'];

  final List<Map<String, dynamic>> _topics = [
    {
      'icon': Icons.functions,
      'title': 'Mathematics',
      'description': 'Explore calculus, algebra, and more',
      'color': const Color(0xFF8B5CF6),
      'topics': ['Calculus', 'Linear Algebra', 'Statistics'],
    },
    {
      'icon': Icons.science,
      'title': 'Science',
      'description': 'Physics, chemistry, and biology',
      'color': const Color(0xFF3B82F6),
      'topics': ['Physics', 'Chemistry', 'Biology'],
    },
    {
      'icon': Icons.psychology,
      'title': 'Philosophy',
      'description': 'Critical thinking and ethics',
      'color': const Color(0xFF10B981),
      'topics': ['Logic', 'Ethics', 'Epistemology'],
    },
    {
      'icon': Icons.code,
      'title': 'Programming',
      'description': 'Data structures and algorithms',
      'color': const Color(0xFFF59E0B),
      'topics': ['Algorithms', 'Data Structures', 'Design Patterns'],
    },
    {
      'icon': Icons.history_edu,
      'title': 'History',
      'description': 'World history and civilizations',
      'color': const Color(0xFFEF4444),
      'topics': ['Ancient', 'Modern', 'World Wars'],
    },
    {
      'icon': Icons.language,
      'title': 'Language Arts',
      'description': 'Writing and literature analysis',
      'color': const Color(0xFF06B6D4),
      'topics': ['Grammar', 'Literature', 'Writing'],
    },
  ];

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return Container(
      decoration: BoxDecoration(
        gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
      ),
      child: CustomScrollView(
        slivers: [
          // Search Header
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Explore Topics',
                    style: Theme.of(context).textTheme.headlineLarge,
                  ),
                  const SizedBox(height: 16),
                  
                  // Search Bar
                  Container(
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceCard,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: AppTheme.primaryLight.withOpacity(0.2),
                      ),
                    ),
                    child: TextField(
                      controller: _searchController,
                      style: const TextStyle(color: AppTheme.textPrimary),
                      decoration: InputDecoration(
                        hintText: 'Search topics...',
                        hintStyle: const TextStyle(color: AppTheme.textMuted),
                        prefixIcon: const Icon(Icons.search, color: AppTheme.textMuted),
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 14,
                        ),
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 20),
                  
                  // Filter Tabs
                  SizedBox(
                    height: 40,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: _tabs.length,
                      itemBuilder: (context, index) {
                        final isSelected = _selectedTab == index;
                        return GestureDetector(
                          onTap: () => setState(() => _selectedTab = index),
                          child: Container(
                            margin: const EdgeInsets.only(right: 12),
                            padding: const EdgeInsets.symmetric(horizontal: 20),
                            decoration: BoxDecoration(
                              color: isSelected 
                                ? AppTheme.accentOrange 
                                : AppTheme.surfaceCard,
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                color: isSelected 
                                  ? AppTheme.accentOrange 
                                  : AppTheme.primaryLight.withOpacity(0.2),
                              ),
                            ),
                            alignment: Alignment.center,
                            child: Text(
                              _tabs[index],
                              style: TextStyle(
                                color: isSelected 
                                  ? Colors.white 
                                  : AppTheme.textSecondary,
                                fontWeight: isSelected 
                                  ? FontWeight.w600 
                                  : FontWeight.normal,
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Topics Grid
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            sliver: SliverList(
              delegate: SliverChildBuilderDelegate(
                (context, index) => _buildTopicCard(_topics[index]),
                childCount: _topics.length,
              ),
            ),
          ),
          
          const SliverToBoxAdapter(
            child: SizedBox(height: 100),
          ),
        ],
      ),
    );
  }

  Widget _buildTopicCard(Map<String, dynamic> topic) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: GradientCard(
        gradient: const LinearGradient(
          colors: [Color(0xFF1E1B2E), Color(0xFF151226)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderGradient: LinearGradient(
          colors: [
            (topic['color'] as Color).withOpacity(0.5),
            (topic['color'] as Color).withOpacity(0.1),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        child: InkWell(
          onTap: () => _showTopicDetails(topic),
          borderRadius: BorderRadius.circular(20),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                // Icon Container
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    color: (topic['color'] as Color).withOpacity(0.15),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Icon(
                    topic['icon'],
                    color: topic['color'],
                    size: 28,
                  ),
                ),
                const SizedBox(width: 16),
                
                // Text Content
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        topic['title'],
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        topic['description'],
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppTheme.textSecondary,
                        ),
                      ),
                    ],
                  ),
                ),
                
                // Arrow
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: topic['color'],
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.arrow_forward,
                    color: Colors.white,
                    size: 18,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _showTopicDetails(Map<String, dynamic> topic) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.surfaceCard,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.4,
        maxChildSize: 0.9,
        expand: false,
        builder: (context, scrollController) => Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Handle
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: AppTheme.textMuted,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              
              // Header
              Row(
                children: [
                  Container(
                    width: 60,
                    height: 60,
                    decoration: BoxDecoration(
                      color: (topic['color'] as Color).withOpacity(0.15),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Icon(
                      topic['icon'],
                      color: topic['color'],
                      size: 32,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          topic['title'],
                          style: Theme.of(context).textTheme.headlineMedium,
                        ),
                        Text(
                          topic['description'],
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: AppTheme.textSecondary,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 32),
              
              Text(
                'Available Topics',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 16),
              
              // Sub-topics
              Expanded(
                child: ListView.builder(
                  controller: scrollController,
                  itemCount: (topic['topics'] as List).length,
                  itemBuilder: (context, index) {
                    final subTopic = topic['topics'][index];
                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: AppTheme.primaryLight.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: ListTile(
                        title: Text(
                          subTopic,
                          style: const TextStyle(color: AppTheme.textPrimary),
                        ),
                        trailing: Icon(
                          Icons.play_arrow,
                          color: topic['color'],
                        ),
                        onTap: () {
                          Navigator.pop(context);
                          // Navigate to chat with topic
                          Navigator.pushNamed(context, '/chat', arguments: subTopic);
                        },
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

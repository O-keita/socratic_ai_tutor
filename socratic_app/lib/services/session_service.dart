import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';
import '../models/session.dart';

class SessionService {
  static const String _sessionsKey = 'chat_sessions';
  
  static const String _privacyModeKey = 'privacy_mode_enabled';

  // Delete all sessions
  static Future<void> clearAll() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_sessionsKey);
  }

  // Save or update a session
  static Future<void> saveSession(Session session) async {
    final prefs = await SharedPreferences.getInstance();

    // When privacy mode is on, keep session in memory but don't persist
    if (prefs.getBool(_privacyModeKey) ?? false) return;

    final sessionsJson = prefs.getStringList(_sessionsKey) ?? [];
    
    final sessions = sessionsJson
        .map((s) => Session.fromJson(jsonDecode(s)))
        .toList();
    
    final index = sessions.indexWhere((s) => s.id == session.id);
    if (index != -1) {
      sessions[index] = session;
    } else {
      sessions.add(session);
    }
    
    final updatedJson = sessions
        .map((s) => jsonEncode(s.toJson()))
        .toList();
    
    await prefs.setStringList(_sessionsKey, updatedJson);
  }

  // Get all sessions
  static Future<List<Session>> getSessions() async {
    final prefs = await SharedPreferences.getInstance();
    final sessionsJson = prefs.getStringList(_sessionsKey) ?? [];
    
    final sessions = sessionsJson
        .map((s) => Session.fromJson(jsonDecode(s)))
        .toList();
    
    // Sort by last active (newest first)
    sessions.sort((a, b) => b.lastActive.compareTo(a.lastActive));
    return sessions;
  }

  // Get a specific session by ID
  static Future<Session?> getSession(String id) async {
    final sessions = await getSessions();
    try {
      return sessions.firstWhere((s) => s.id == id);
    } catch (e) {
      return null;
    }
  }

  // Delete a session
  static Future<void> deleteSession(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final sessionsJson = prefs.getStringList(_sessionsKey) ?? [];
    
    final sessions = sessionsJson
        .map((s) => Session.fromJson(jsonDecode(s)))
        .toList();
    
    sessions.removeWhere((s) => s.id == id);
    
    final updatedJson = sessions
        .map((s) => jsonEncode(s.toJson()))
        .toList();
    
    await prefs.setStringList(_sessionsKey, updatedJson);
  }

  // Create a new session
  static Session createNewSession({String? topic}) {
    final now = DateTime.now();
    return Session(
      id: const Uuid().v4(),
      topic: topic ?? 'General Discussion',
      messages: [],
      startTime: now,
      lastActive: now,
    );
  }
}

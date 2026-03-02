import { useState, useMemo } from 'react'
import { BookOpen, Search } from 'lucide-react'

interface GlossaryTerm {
  term: string
  definition: string
  category: string
}

// Sample glossary data - in production this would come from the backend
const GLOSSARY_TERMS: GlossaryTerm[] = [
  {
    term: 'Overfitting',
    definition: 'A modeling error that occurs when a function is too closely aligned to a limited set of data points. As a result, the model is useful in reference only to its initial data set and not to any other data sets.',
    category: 'Machine Learning'
  },
  {
    term: 'Underfitting',
    definition: 'A scenario where a data model is unable to capture the relationship between the input and output variables accurately, generating a high error rate on both the training set and unseen data.',
    category: 'Machine Learning'
  },
  {
    term: 'Supervised Learning',
    definition: 'A type of machine learning where the algorithm is trained on a labeled dataset, meaning each training example is paired with an output label.',
    category: 'Machine Learning'
  },
  {
    term: 'Unsupervised Learning',
    definition: 'A type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses.',
    category: 'Machine Learning'
  },
  {
    term: 'Neural Network',
    definition: 'A series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.',
    category: 'Deep Learning'
  },
  {
    term: 'Backpropagation',
    definition: 'The standard method of training artificial neural networks. It calculates the gradient of the loss function with respect to the weights of the network.',
    category: 'Deep Learning'
  },
  {
    term: 'Transformer',
    definition: 'A deep learning model architecture that uses the mechanism of self-attention, weighing the significance of each part of the input data differently.',
    category: 'Natural Language Processing'
  },
  {
    term: 'Fine-Tuning',
    definition: 'The process of taking a pre-trained model and further training it on a specific dataset to adapt it to a particular task or domain.',
    category: 'Machine Learning'
  },
  {
    term: 'Gradient Descent',
    definition: 'An optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient.',
    category: 'Optimization'
  },
  {
    term: 'LSTM',
    definition: 'Long Short-Term Memory networks are a type of recurrent neural network capable of learning long-term dependencies, especially useful in sequence prediction problems.',
    category: 'Deep Learning'
  }
]

export default function GlossaryPage() {
  const [search, setSearch] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('All')

  const categories = useMemo(() => {
    const cats = [...new Set(GLOSSARY_TERMS.map(t => t.category))].sort()
    return ['All', ...cats]
  }, [])

  const filtered = useMemo(() => {
    let terms = GLOSSARY_TERMS
    if (selectedCategory !== 'All') {
      terms = terms.filter(t => t.category === selectedCategory)
    }
    if (search) {
      const q = search.toLowerCase()
      terms = terms.filter(
        t => t.term.toLowerCase().includes(q) || t.definition.toLowerCase().includes(q)
      )
    }
    return terms.sort((a, b) => a.term.localeCompare(b.term))
  }, [search, selectedCategory])

  return (
    <div className="max-w-5xl mx-auto p-4 md:p-8 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-500 to-teal-500 rounded-2xl p-6 md:p-8 text-white">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-white/20 flex items-center justify-center">
            <BookOpen className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold">Glossary</h1>
            <p className="text-emerald-100 text-sm md:text-base">
              Key terms in ML & Data Science
            </p>
          </div>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
        <input
          type="text"
          placeholder="Search terms..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-12 pr-4 py-3 rounded-xl border border-slate-200 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 transition-all"
        />
      </div>

      {/* Category Filter */}
      <div className="flex flex-wrap gap-2">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedCategory === cat
                ? 'bg-emerald-500 text-white'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Results */}
      <div className="text-sm text-slate-500 font-medium">
        {filtered.length} {filtered.length === 1 ? 'term' : 'terms'}
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-12 text-slate-400">
          <BookOpen className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="font-medium text-slate-600">No terms found</p>
          <button
            onClick={() => { setSearch(''); setSelectedCategory('All') }}
            className="text-sm text-emerald-500 hover:text-emerald-600 mt-2 font-medium"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {filtered.map((term, idx) => (
            <div
              key={idx}
              className="bg-white rounded-xl border border-slate-100 shadow-sm p-5 hover:shadow-md hover:border-emerald-200 transition-all"
            >
              <div className="flex items-start gap-3 mb-2">
                <div className="flex-1">
                  <h3 className="font-bold text-slate-800 text-lg">{term.term}</h3>
                  <span className="inline-block mt-1 px-2 py-0.5 bg-emerald-50 text-emerald-600 text-xs font-semibold rounded">
                    {term.category}
                  </span>
                </div>
              </div>
              <p className="text-sm text-slate-600 leading-relaxed mt-3">{term.definition}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

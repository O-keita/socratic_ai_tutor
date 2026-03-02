import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import type { Components } from 'react-markdown'

interface LessonReaderProps {
  content: string
}

const components: Components = {
  code({ node, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '')
    const inline = !match
    if (inline) {
      return (
        <code
          className="bg-orange-50 text-orange-600 px-1.5 py-0.5 rounded text-sm font-mono border border-orange-100"
          {...props}
        >
          {children}
        </code>
      )
    }
    return (
      <SyntaxHighlighter
        style={oneDark}
        language={match[1]}
        PreTag="div"
        className="rounded-xl text-sm"
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    )
  },
}

export default function LessonReader({ content }: LessonReaderProps) {
  return (
    <div className="prose prose-slate max-w-none prose-headings:text-slate-800 prose-p:text-slate-600 prose-a:text-orange-500 prose-strong:text-slate-700">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

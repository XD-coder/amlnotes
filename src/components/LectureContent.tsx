"use client";

import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkTable from "remark-gfm";

interface LectureContentProps {
  content: string;
  title: string;
}

export default function LectureContent({
  content,
  title,
}: LectureContentProps) {
  return (
    <div className="w-full">
      <div className="mb-8 pb-6 border-b-2 border-gray-200">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">{title}</h1>
        <p className="text-gray-500">Applied Machine Learning Study Notes</p>
      </div>

      <article className="prose prose-lg max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkTable]}
          components={{
            h2: ({ children }) => (
              <h2 className="mt-8 mb-4 text-2xl font-bold text-gray-900 border-l-4 border-blue-600 pl-4">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="mt-6 mb-3 text-xl font-bold text-gray-800">
                {children}
              </h3>
            ),
            h4: ({ children }) => (
              <h4 className="mt-4 mb-2 text-lg font-semibold text-gray-700">
                {children}
              </h4>
            ),
            p: ({ children }) => (
              <p className="my-4 text-gray-700 leading-7">{children}</p>
            ),
            table: ({ children }) => (
              <div className="my-6 overflow-x-auto">
                <table className="min-w-full border-collapse border border-gray-300">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-blue-100">{children}</thead>
            ),
            tbody: ({ children }) => (
              <tbody className="bg-white">{children}</tbody>
            ),
            tr: ({ children }) => (
              <tr className="border border-gray-300">{children}</tr>
            ),
            th: ({ children }) => (
              <th className="px-4 py-2 text-left font-semibold text-gray-800 border border-gray-300">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-4 py-2 text-gray-700 border border-gray-300">
                {children}
              </td>
            ),
            ul: ({ children }) => (
              <ul className="my-4 ml-6 space-y-2 text-gray-700 list-disc">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="my-4 ml-6 space-y-2 text-gray-700 list-decimal">
                {children}
              </ol>
            ),
            li: ({ children }) => <li className="my-1">{children}</li>,
            code: ({ inline, children }: { inline?: boolean; children?: React.ReactNode }) =>
              inline ? (
                <code className="bg-gray-100 text-red-600 px-2 py-1 rounded font-mono text-sm">
                  {children}
                </code>
              ) : (
                <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto my-4">
                  <code className="text-gray-800 font-mono text-sm">
                    {children}
                  </code>
                </pre>
              ),
            blockquote: ({ children }) => (
              <blockquote className="my-4 pl-4 border-l-4 border-blue-500 text-gray-600 italic">
                {children}
              </blockquote>
            ),
            strong: ({ children }) => (
              <strong className="font-bold text-gray-900">{children}</strong>
            ),
            em: ({ children }) => (
              <em className="italic text-gray-700">{children}</em>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      </article>
    </div>
  );
}

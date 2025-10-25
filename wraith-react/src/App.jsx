import React, { useEffect, useState } from 'react'

export default function App(){
  const [url, setUrl] = useState('http://localhost:8000/index.html')
  const [status, setStatus] = useState('Checking…')

  useEffect(() => {
    // Probe the static server quickly
    const ctrl = new AbortController()
    fetch('http://localhost:8000/', { signal: ctrl.signal })
      .then(() => setStatus('Static server: OK (8000)'))
      .catch(() => setStatus('Static server not found on 8000 — start it or change URL below.'))
    return () => ctrl.abort()
  }, [])

  return (
    <div style={{height:'100%', display:'flex', flexDirection:'column'}}>
      <div className="bar">
        <strong>Wraith React Host</strong>
        <span style={{marginLeft:8}}>{status}</span>
        <span style={{marginLeft:'auto'}}>
          Iframe URL:
          <input style={{marginLeft:6, width:420}} value={url} onChange={e=>setUrl(e.target.value)} />
          <a href={url} target="_blank" rel="noreferrer" style={{marginLeft:8}}>open</a>
        </span>
      </div>
      <iframe className="frame" title="wraith" src={url} />
    </div>
  )
}

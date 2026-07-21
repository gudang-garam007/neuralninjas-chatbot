-- Run this in Supabase SQL Editor before first use

create extension if not exists vector;

create table if not exists nn_documents (
    id bigserial primary key,
    url text not null,
    title text,
    content text not null,
    embedding vector(384),  -- matches BAAI/bge-small-en-v1.5 dims
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- similarity search index
create index if not exists nn_documents_embedding_idx
    on nn_documents using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create index if not exists nn_documents_url_idx on nn_documents (url);

-- similarity search function used by the backend
create or replace function match_nn_documents (
    query_embedding vector(384),
    match_threshold float default 0.5,
    match_count int default 5
)
returns table (
    id bigint,
    url text,
    title text,
    content text,
    similarity float
)
language sql stable
as $$
    select
        nn_documents.id,
        nn_documents.url,
        nn_documents.title,
        nn_documents.content,
        1 - (nn_documents.embedding <=> query_embedding) as similarity
    from nn_documents
    where 1 - (nn_documents.embedding <=> query_embedding) > match_threshold
    order by nn_documents.embedding <=> query_embedding
    limit match_count;
$$;

-- optional: simple session memory table
create table if not exists nn_chat_sessions (
    session_id text primary key,
    messages jsonb default '[]'::jsonb,
    updated_at timestamptz default now()
);

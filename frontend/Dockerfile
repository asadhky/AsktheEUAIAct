# The chat page. Build from this directory:   docker build -t askact-ui frontend
#
# NEXT_PUBLIC_API_URL is a BUILD-TIME value. Next.js writes it into the JavaScript the browser downloads when the page is
# built, so changing it when the container starts has no effect: pass it as a build argument, and rebuild to change it.
# It must be the address of the backend as the BROWSER sees it (the visitor's machine, not another container). For a
# deployment that is the backend's public URL, and that page's own origin must be in the backend's CORS_ORIGINS.

FROM node:22-slim AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json next.config.mjs ./
COPY app app
COPY lib lib
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
RUN npm run build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000
# The standalone server, and the static files it does not copy for itself. The `node` user is not root.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]

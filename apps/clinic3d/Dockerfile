FROM node:22-alpine
WORKDIR /app
COPY --chown=node:node . .
USER node
ENV NODE_ENV=production
ENV HOST=0.0.0.0
EXPOSE 8090
CMD ["node","server.mjs"]

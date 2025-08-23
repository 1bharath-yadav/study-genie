# Use an official Node.js image as the base
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY ./package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Build the frontend
RUN npm run build

# Use nginx to serve the build
FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
COPY ./nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

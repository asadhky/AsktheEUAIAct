/** @type {import('next').NextConfig} */
const nextConfig = {
  // Makes `next build` write a self-contained server (.next/standalone) that carries only the files it needs, so the
  // Docker image does not have to ship node_modules. It changes nothing for `next dev` or the tests.
  output: "standalone",
};

export default nextConfig;

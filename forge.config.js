const path = require('path');
const fs = require('fs-extra');

module.exports = {  packagerConfig: {
    ignore: [
      /\/models\.json$/,
      /\/python_backend\/models\.json$/
    ]
  },
  hooks: {
    postMake: async (forgeConfig, options) => {
      const outputPath = options.outputPaths[0];
      await fs.copy(
        path.resolve(__dirname, 'python_backend/models.json'),
        path.join(outputPath, 'models.json')
      );
    }
  }
};